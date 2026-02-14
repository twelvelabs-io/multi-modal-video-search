"""
Chat Agent for the Analyze page.

Uses Bedrock Converse API with Claude to reason about user intent
and orchestrate tools: search (segments/assets), Pegasus analysis,
subclip creation, and concatenation via Lambda + FFmpeg.
"""

import json
import logging
import os
import re
from urllib.parse import urlparse

import boto3

logger = logging.getLogger(__name__)

# Claude 3 Haiku — confirmed working in this account (used by decompose_query)
AGENT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

TOOL_DEFINITIONS = [
    {
        "toolSpec": {
            "name": "search_segments",
            "description": (
                "Search for specific video clips, segments, scenes, or moments "
                "using Marengo multimodal embeddings. Returns ranked segments with "
                "timestamps and confidence scores. Use when the user wants to find "
                "specific clips, scenes, or moments."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Clean search query. Remove conversational phrasing like 'find me', 'show me', 'give me the best', etc."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results to return. Extract from query if specified (e.g. 'find 5 clips' -> 5). Default 10."
                        },
                        "video_id": {
                            "type": "string",
                            "description": "Video ID to filter results to a specific video. Use the selected video's ID if provided in context."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "search_assets",
            "description": (
                "Search for full videos or assets (not individual segments). "
                "Aggregates segment scores by video and returns one result per video "
                "with the best-matching segment info. Use when the user says "
                "'find videos', 'find assets', 'find episodes', or wants video-level results."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Clean search query."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of videos to return. Default 5."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "analyze_video",
            "description": (
                "Analyze a video or video segment using TwelveLabs Pegasus. "
                "Answers natural language questions about video content: what is happening, "
                "who is speaking, describe the scene, etc. Use when the user asks about "
                "the content of a specific video or quoted segment. "
                "Constraint: video must be under 1 hour."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "s3_uri": {
                            "type": "string",
                            "description": "S3 URI of the video (from context or search results)."
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Natural language question about the video content."
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Segment start time in seconds (from quoted segment)."
                        },
                        "end_time": {
                            "type": "number",
                            "description": "Segment end time in seconds (from quoted segment)."
                        }
                    },
                    "required": ["s3_uri", "prompt"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "create_subclip",
            "description": (
                "Create a subclip from a video using FFmpeg. Use this to extract a time range "
                "before calling analyze_video on a quoted segment. The output is a new S3 URI "
                "that can be passed to analyze_video. Max duration: 1 hour."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "s3_uri": {"type": "string", "description": "S3 URI of the source video."},
                        "start_time": {"type": "number", "description": "Start time in seconds."},
                        "end_time": {"type": "number", "description": "End time in seconds."}
                    },
                    "required": ["s3_uri", "start_time", "end_time"]
                }
            }
        }
    },
    {
        "toolSpec": {
            "name": "concatenate_clips",
            "description": (
                "Merge multiple video subclips into a single clip using FFmpeg. "
                "Use when the user has quoted multiple segments and wants them analyzed together. "
                "Returns a new S3 URI of the combined clip. Max 10 clips."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "clips": {
                            "type": "array",
                            "description": "List of clips to concatenate.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "s3_uri": {"type": "string"},
                                    "start_time": {"type": "number"},
                                    "end_time": {"type": "number"}
                                }
                            }
                        }
                    },
                    "required": ["clips"]
                }
            }
        }
    }
]


SUBCLIP_LAMBDA_NAME = "video-subclip-processor"

OLD_S3_PREFIX = "s3://tl-brice-media/WBD_project/Videos/proxy/"
NEW_S3_PREFIX = "s3://multi-modal-video-search-app/proxies/"

# Keywords that indicate the user wants to SEARCH, not analyze a selected video
SEARCH_KEYWORDS = ["find", "search", "look for", "show me clips", "find me", "find videos"]

# Keywords/patterns that indicate ASSET-level search (not segment-level)
# Exact substring matches checked first (fast)
ASSET_KEYWORDS = [
    "find asset", "find assets", "find video", "find videos",
    "find episode", "find episodes", "search asset", "search assets",
    "search video", "search videos", "show me video", "show me videos",
]

# Regex patterns for natural language asset requests (checked second)
# Matches: "find me the lexus commercial videos", "find all videos with lexus", etc.
ASSET_PATTERNS = [
    re.compile(r"\b(?:find|search|show)\b.*\b(?:asset|assets|video|videos|episode|episodes)\b", re.IGNORECASE),
]

# Regex to strip asset/search preamble and extract the clean query
ASSET_QUERY_STRIP = re.compile(
    r"^(?:find|search|show me|look for)\s+(?:\w+\s+)*?(?:asset|assets|video|videos|episode|episodes)\s+"
    r"(?:of|about|with|for|featuring|containing|related to|on|showing)?\s*",
    re.IGNORECASE
)

# Keywords that indicate the user wants a highlight reel
HIGHLIGHT_KEYWORDS = ["highlight", "best moments", "highlight reel", "key moments", "recap", "montage"]

HIGHLIGHT_PEGASUS_PROMPT = (
    "Identify the 5-8 most important or memorable moments in this video. "
    "You MUST respond using EXACTLY this format for each moment:\n\n"
    "MOMENT [start]-[end]: description\n\n"
    "Where [start] and [end] are times in SECONDS (e.g. 10.0, 65.5, 120.0).\n"
    "Each clip should be 4-10 seconds.\n\n"
    "Example output:\n"
    "MOMENT 10.0-18.0: The host introduces the main topic\n"
    "MOMENT 45.0-53.0: A surprising reveal happens\n"
    "MOMENT 120.0-130.0: The climactic scene\n\n"
    "IMPORTANT: You MUST use the MOMENT format above. Do NOT write prose paragraphs."
)

# Fallback prompt for Claude to extract timestamps from Pegasus prose
CLAUDE_TIMESTAMP_EXTRACTION_PROMPT = """You are extracting highlight moments from a video description. The video is approximately {duration_minutes:.0f} minutes ({duration_seconds:.0f} seconds).

VIDEO DESCRIPTION:
{description}

Output ONLY a JSON array with 5-8 highlight moments. Each moment is 5-12 seconds long.

[{{"title": "Brief description", "start_time": 10.0, "end_time": 18.0}}]

Timestamp estimation rules:
- If the description mentions specific times (e.g. "at 1:30"), convert to seconds
- Otherwise estimate proportionally: opening=5-10%, early=10-25%, midpoint=40-60%, late=70-85%, climax=85-95%
- Spread moments evenly across the video — don't cluster them
- Each clip: start_time and end_time in seconds, end_time = start_time + 5 to 12
- Order chronologically

Output ONLY the JSON array, no other text."""


# Estimated cost per API call (USD) — approximate Bedrock marketplace pricing
COST_PER_CALL = {
    "marengo_search": 0.012,   # Marengo 3.0 embedding + vector lookup
    "pegasus_analysis": 0.05,  # Pegasus 1.2 video understanding (approx per call)
    "claude_haiku": 0.001,     # Claude 3 Haiku agent reasoning turn
    "aws_lambda": 0.002,       # Lambda compute (1024MB, <10s typical)
}


class ChatAgent:
    """Bedrock Converse API agent for the Analyze chat page."""

    def __init__(
        self,
        bedrock_runtime_client,
        search_client,
        cloudfront_domain: str,
        model_id: str = AGENT_MODEL_ID,
        max_turns: int = 5
    ):
        self.client = bedrock_runtime_client
        self.search_client = search_client
        self.cloudfront_domain = cloudfront_domain
        self.model_id = model_id
        self.max_turns = max_turns
        self.lambda_client = boto3.client("lambda", region_name="us-east-1")

    def run(self, message: str, chat_history: list, context: dict) -> dict:
        """
        Run the agent loop.

        Args:
            message: User's current message
            chat_history: Previous messages [{role, content}]
            context: {selected_video, quoted_segment, settings}

        Returns:
            {message: str, actions: list, tool_calls: list}
        """
        # Direct pipeline for quoted segments — bypass LLM to guarantee
        # subclip/concat → analyze_video chaining
        quoted_segments = context.get("quoted_segments", [])
        if context.get("quoted_segment") and not quoted_segments:
            quoted_segments = [context["quoted_segment"]]
        if quoted_segments:
            return self._run_quoted_pipeline(message, quoted_segments)

        # Direct pipeline for selected video content questions — bypass LLM
        # because Haiku unreliably follows system prompt for analyze_video calls
        if not self._is_search_request(message):
            selected_videos = context.get("selected_videos", [])
            if not selected_videos and context.get("selected_video"):
                selected_videos = [context["selected_video"]]
            videos_with_uri = [v for v in selected_videos if v.get("s3_uri")]
            if videos_with_uri:
                # Highlight pipeline: Pegasus identifies moments → FFmpeg concatenates
                if self._is_highlight_request(message):
                    return self._run_highlight_pipeline(message, videos_with_uri)
                return self._run_video_analysis(message, videos_with_uri)

        # Direct pipeline for asset-level searches — bypass LLM because Haiku
        # unreliably distinguishes search_assets from search_segments
        settings = context.get("settings", {})
        if context.get("query_image"):
            settings["query_image"] = context["query_image"]

        # Direct image search — bypass LLM when image is provided
        if settings.get("query_image"):
            return self._run_image_search(message, settings)

        if self._is_asset_search_request(message):
            return self._run_asset_search(message, settings)

        system_prompt = self._build_system_prompt(context)
        messages = self._build_messages(chat_history, message)

        actions = []
        tool_calls = []
        final_text = ""
        agent_turns = 0

        for turn in range(self.max_turns):
            agent_turns += 1
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    system=[{"text": system_prompt}],
                    messages=messages,
                    toolConfig={"tools": TOOL_DEFINITIONS},
                    inferenceConfig={"maxTokens": 1024, "temperature": 0.3}
                )
            except Exception as e:
                logger.error(f"Converse API error: {e}")
                return {"message": f"Agent error: {str(e)}", "actions": [], "tool_calls": []}

            output = response.get("output", {})
            content_blocks = output.get("message", {}).get("content", [])
            stop_reason = response.get("stopReason", "end_turn")

            # Append assistant response to conversation
            messages.append({"role": "assistant", "content": content_blocks})

            if stop_reason == "tool_use":
                tool_results = []
                for block in content_blocks:
                    if "toolUse" in block:
                        tool_use = block["toolUse"]
                        tool_name = tool_use["name"]
                        tool_id = tool_use["toolUseId"]
                        tool_input = tool_use["input"]

                        logger.info(f"Agent tool call: {tool_name}({json.dumps(tool_input, default=str)})")

                        try:
                            result = self._execute_tool(tool_name, tool_input, settings)
                            actions.append(result)
                            tool_calls.append({
                                "name": tool_name,
                                "input": tool_input,
                                "output_summary": self._summarize_result(result)
                            })
                            tool_results.append({
                                "toolUseId": tool_id,
                                "content": [{"json": result}]
                            })
                        except Exception as e:
                            logger.error(f"Tool error ({tool_name}): {e}")
                            tool_results.append({
                                "toolUseId": tool_id,
                                "content": [{"text": f"Error: {str(e)}"}],
                                "status": "error"
                            })

                # Feed tool results back
                messages.append({
                    "role": "user",
                    "content": [{"toolResult": tr} for tr in tool_results]
                })
                continue

            # No tool use — extract final text
            for block in content_blocks:
                if "text" in block:
                    final_text += block["text"]
            break

        # Add Claude Haiku agent turns to tool_calls for cost tracking
        for _ in range(agent_turns):
            tool_calls.append({"name": "claude_agent_turn", "input": {}, "output_summary": "Agent reasoning"})
        costs = self._calculate_costs(tool_calls)
        return {"message": final_text, "actions": actions, "tool_calls": tool_calls, "costs": costs}

    # ── Helper: detect search intent ──

    @staticmethod
    def _is_search_request(message: str) -> bool:
        """Check if the message is a search request vs. a content question."""
        msg_lower = message.lower().strip()
        return any(kw in msg_lower for kw in SEARCH_KEYWORDS)

    @staticmethod
    def _is_highlight_request(message: str) -> bool:
        """Check if the user wants a highlight reel / best moments compilation."""
        msg_lower = message.lower().strip()
        return any(kw in msg_lower for kw in HIGHLIGHT_KEYWORDS)

    @staticmethod
    def _is_asset_search_request(message: str) -> bool:
        """Check if the user wants video/asset-level search (not segment-level)."""
        msg_lower = message.lower().strip()
        # Fast check: exact keyword match
        if any(kw in msg_lower for kw in ASSET_KEYWORDS):
            return True
        # Regex check: natural language patterns like "find me the X videos"
        return any(p.search(msg_lower) for p in ASSET_PATTERNS)

    @staticmethod
    def _extract_clean_query(message: str) -> str:
        """Strip search preamble to get the core query text."""
        clean = ASSET_QUERY_STRIP.sub("", message).strip()
        return clean if clean else message.strip()

    def _run_image_search(self, message: str, settings: dict) -> dict:
        """Direct image search — bypass the LLM agent when image is provided."""
        query = message.strip() if message and message.strip().lower() != 'image search' else ''
        logger.info(f"Direct image search: query='{query}', has_image=True")

        tool_input = {"query": query, "limit": 20}
        result = self._exec_search_segments(tool_input, settings)
        tool_calls = [{
            "name": "search_segments",
            "input": {"query": query or "(image)", "limit": 20},
            "output_summary": self._summarize_result(result)
        }]
        costs = self._calculate_costs(tool_calls)
        n = len(result.get("results", []))
        label = f'"{query}" + image' if query else "image"
        return {
            "message": f"Found {n} result{'s' if n != 1 else ''} matching {label}:",
            "actions": [result],
            "tool_calls": tool_calls,
            "costs": costs
        }

    def _run_asset_search(self, message: str, settings: dict) -> dict:
        """Direct asset-level search — bypass the LLM agent for reliability."""
        query = self._extract_clean_query(message)
        logger.info(f"Direct asset search: query='{query}' (from: '{message}')")

        tool_input = {"query": query, "limit": 5}
        result = self._exec_search_assets(tool_input, settings)
        tool_calls = [{
            "name": "search_assets",
            "input": tool_input,
            "output_summary": self._summarize_result(result)
        }]
        costs = self._calculate_costs(tool_calls)
        n = len(result.get("results", []))
        return {
            "message": f"Found {n} video{'s' if n != 1 else ''} matching \"{query}\":",
            "actions": [result],
            "tool_calls": tool_calls,
            "costs": costs
        }

    # ── Highlight Video Pipeline ──

    def _run_highlight_pipeline(self, message: str, videos: list) -> dict:
        """
        Generate a highlight video:
        1. Ask Pegasus to identify key moments with timestamps
        2. Concatenate those moments into a highlight reel via FFmpeg Lambda
        """
        actions = []
        tool_calls = []

        vid = videos[0]  # Use first selected video
        s3_uri = self._normalize_s3_uri(vid.get("s3_uri", ""))
        name = vid.get("name", "Unknown")
        # Use actual duration if available, else estimate from segment count
        video_duration = vid.get("duration", 0)
        if not video_duration:
            seg_count = vid.get("segment_count", vid.get("segments", 0))
            if isinstance(seg_count, int) and seg_count > 0:
                video_duration = seg_count * 6  # ~6s per segment estimate
            else:
                video_duration = 600  # 10 min default

        # Step 1: Ask Pegasus for highlight timestamps
        logger.info(f"Highlight pipeline: asking Pegasus for key moments in {name}")
        analyze_input = {"s3_uri": s3_uri, "prompt": HIGHLIGHT_PEGASUS_PROMPT}
        try:
            analysis_result = self._exec_analyze_video(analyze_input)
            tool_calls.append({
                "name": "analyze_video",
                "input": {"s3_uri": s3_uri, "prompt": "Identify highlight moments with timestamps"},
                "output_summary": self._summarize_result(analysis_result)
            })
        except Exception as e:
            error_msg = f"Pegasus highlight analysis of {name} failed: {str(e)}"
            logger.error(error_msg)
            actions.append({"type": "error", "message": error_msg})
            costs = self._calculate_costs(tool_calls)
            return {"message": "", "actions": actions, "tool_calls": tool_calls, "costs": costs}

        # Step 2: Parse timestamps from Pegasus response
        pegasus_text = analysis_result.get("text", "")
        highlights = self._parse_highlight_timestamps(pegasus_text)

        if not highlights:
            # Pegasus returned prose — use Claude to extract timestamps
            logger.info("Pegasus returned prose, falling back to Claude for timestamp extraction")
            highlights = self._extract_timestamps_with_claude(pegasus_text, video_duration)
            if highlights:
                tool_calls.append({
                    "name": "claude_timestamp_extraction",
                    "input": {"description_length": len(pegasus_text)},
                    "output_summary": f"Extracted {len(highlights)} moments from Pegasus prose"
                })

        if not highlights:
            # Neither parser worked — show the raw analysis
            actions.append(analysis_result)
            actions.append({
                "type": "error",
                "message": "Could not extract timestamps from Pegasus response. Showing raw analysis above."
            })
            costs = self._calculate_costs(tool_calls)
            return {"message": "", "actions": actions, "tool_calls": tool_calls, "costs": costs}

        logger.info(f"Highlight pipeline: extracted {len(highlights)} moments")

        # Step 3: Concatenate highlights via FFmpeg Lambda
        clips = [
            {"s3_uri": s3_uri, "start_time": h["start_time"], "end_time": h["end_time"]}
            for h in highlights
        ]
        try:
            concat_result = self._exec_concatenate_clips({"clips": clips})
            tool_calls.append({
                "name": "concatenate_clips",
                "input": {"clips": clips},
                "output_summary": self._summarize_result(concat_result)
            })

            if concat_result.get("type") != "error":
                actions.append(concat_result)
                # Build summary of highlights
                summary_lines = [f"Highlight reel for **{name}** ({len(highlights)} moments):"]
                for i, h in enumerate(highlights):
                    title = h.get("title", f"Moment {i+1}")
                    summary_lines.append(
                        f"{i+1}. {title} ({h['start_time']:.1f}s - {h['end_time']:.1f}s)"
                    )
                costs = self._calculate_costs(tool_calls)
                return {
                    "message": "\n".join(summary_lines),
                    "actions": actions,
                    "tool_calls": tool_calls,
                    "costs": costs
                }
            else:
                actions.append(concat_result)
        except Exception as e:
            error_msg = f"Highlight concatenation failed: {str(e)}"
            logger.error(error_msg)
            actions.append({"type": "error", "message": error_msg})

        costs = self._calculate_costs(tool_calls)
        return {"message": "", "actions": actions, "tool_calls": tool_calls, "costs": costs}

    @staticmethod
    def _parse_timestamp_seconds(ts: str) -> float:
        """Convert a timestamp string (seconds, MM:SS, or HH:MM:SS) to seconds."""
        ts = ts.strip()
        parts = ts.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(ts)

    @staticmethod
    def _parse_highlight_timestamps(text: str) -> list:
        """Parse timestamps from Pegasus response — tries JSON, MOMENT format, and MM:SS patterns."""

        def _validate(start, end):
            return end > start and (end - start) <= 120

        # Try 1: JSON array
        json_match = re.search(r'\[[\s\S]*?\]', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    highlights = []
                    for item in data:
                        if isinstance(item, dict) and "start_time" in item and "end_time" in item:
                            start = float(item["start_time"])
                            end = float(item["end_time"])
                            if _validate(start, end):
                                highlights.append({
                                    "title": item.get("title", item.get("description", "")),
                                    "start_time": start,
                                    "end_time": end
                                })
                    if highlights:
                        return highlights
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Try 2: "MOMENT start-end: description" line format
        moment_pattern = re.findall(
            r'MOMENT\s+([\d.:]+)\s*[-–—to]+\s*([\d.:]+)\s*[:\-–—]\s*(.+)',
            text, re.IGNORECASE
        )
        if moment_pattern:
            highlights = []
            for start_str, end_str, title in moment_pattern:
                try:
                    start = ChatAgent._parse_timestamp_seconds(start_str)
                    end = ChatAgent._parse_timestamp_seconds(end_str)
                    if _validate(start, end):
                        highlights.append({"title": title.strip(), "start_time": start, "end_time": end})
                except (ValueError, IndexError):
                    continue
            if highlights:
                return highlights

        # Try 3: MM:SS-MM:SS or HH:MM:SS-HH:MM:SS patterns in any context
        time_range_pattern = re.findall(
            r'(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—to]+\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*[:\-–—]?\s*(.+?)(?:\n|$)',
            text
        )
        if time_range_pattern:
            highlights = []
            for start_str, end_str, title in time_range_pattern:
                try:
                    start = ChatAgent._parse_timestamp_seconds(start_str)
                    end = ChatAgent._parse_timestamp_seconds(end_str)
                    if _validate(start, end):
                        highlights.append({"title": title.strip(), "start_time": start, "end_time": end})
                except (ValueError, IndexError):
                    continue
            if highlights:
                return highlights

        # Try 4: Numbered list with second-based ranges like "10.0-18.0" or "10-18"
        numbered_pattern = re.findall(
            r'(?:^|\n)\s*\d+[.)]\s*([\d.]+)\s*[-–—]\s*([\d.]+)\s*(?:s(?:ec(?:onds?)?)?)?\s*[:\-–—]\s*(.+?)(?:\n|$)',
            text
        )
        if numbered_pattern:
            highlights = []
            for start_str, end_str, title in numbered_pattern:
                try:
                    start = float(start_str)
                    end = float(end_str)
                    if _validate(start, end):
                        highlights.append({"title": title.strip(), "start_time": start, "end_time": end})
                except (ValueError, IndexError):
                    continue
            if highlights:
                return highlights

        return []

    def _extract_timestamps_with_claude(self, pegasus_prose: str, video_duration: float = 1200) -> list:
        """Use Claude Haiku to extract timestamps from Pegasus prose description."""
        prompt = CLAUDE_TIMESTAMP_EXTRACTION_PROMPT.format(
            description=pegasus_prose,
            duration_minutes=video_duration / 60,
            duration_seconds=video_duration
        )
        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": "You are a JSON timestamp extractor. Output ONLY valid JSON arrays."}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 512, "temperature": 0.1}
            )
            output_text = ""
            for block in response.get("output", {}).get("message", {}).get("content", []):
                if "text" in block:
                    output_text += block["text"]

            logger.info(f"Claude timestamp extraction response: {output_text[:200]}")
            return self._parse_highlight_timestamps(output_text)
        except Exception as e:
            logger.error(f"Claude timestamp extraction failed: {e}")
            return []

    @staticmethod
    def _calculate_costs(tool_calls: list) -> dict:
        """Calculate estimated API costs from tool calls."""
        breakdown = []
        total = 0.0
        for tc in tool_calls:
            name = tc.get("name", "")
            if name in ("search_segments", "search_assets"):
                cost = COST_PER_CALL["marengo_search"]
                label = "Marengo 3.0"
            elif name == "analyze_video":
                cost = COST_PER_CALL["pegasus_analysis"]
                label = "Pegasus 1.2"
            elif name in ("create_subclip", "concatenate_clips"):
                cost = COST_PER_CALL["aws_lambda"]
                label = "AWS Lambda"
            elif name in ("claude_timestamp_extraction", "claude_agent_turn"):
                cost = COST_PER_CALL["claude_haiku"]
                label = "Claude Haiku"
            else:
                continue
            total += cost
            breakdown.append({"tool": name, "label": label, "cost": cost})

        return {"total": round(total, 4), "breakdown": breakdown}

    # ── Selected Video Analysis Pipeline ──

    def _run_video_analysis(self, message: str, videos: list) -> dict:
        """
        Directly analyze selected videos with Pegasus — bypass the LLM agent.

        Called when videos are selected and the user asks a content question
        (anything that's not a search request).
        """
        actions = []
        tool_calls = []

        for vid in videos:
            s3_uri = self._normalize_s3_uri(vid.get("s3_uri", ""))
            name = vid.get("name", "Unknown")
            analyze_input = {"s3_uri": s3_uri, "prompt": message}

            logger.info(f"Direct video analysis: {name} ({s3_uri})")
            try:
                result = self._exec_analyze_video(analyze_input)
                actions.append(result)
                tool_calls.append({
                    "name": "analyze_video",
                    "input": analyze_input,
                    "output_summary": self._summarize_result(result)
                })
            except Exception as e:
                error_msg = f"Pegasus analysis of {name} failed: {str(e)}"
                logger.error(error_msg)
                actions.append({"type": "error", "message": error_msg})
                tool_calls.append({
                    "name": "analyze_video",
                    "input": analyze_input,
                    "output_summary": f"ERROR: {error_msg}"
                })

        costs = self._calculate_costs(tool_calls)
        return {"message": "", "actions": actions, "tool_calls": tool_calls, "costs": costs}

    # ── Quoted Segments Pipeline ──

    def _run_quoted_pipeline(self, message: str, quoted_segments: list) -> dict:
        """
        Handle selected segments: create clip via Lambda, then analyze clip with Pegasus.

        1. Create subclip/concatenation via Lambda (FFmpeg)
        2. Analyze the CREATED CLIP with Pegasus (not the original video)
        3. If clip creation fails, fall back to original video + time ranges
        """
        actions = []
        tool_calls = []
        clip_s3_uri = None

        # Step 1: Create clip via Lambda
        try:
            if len(quoted_segments) == 1:
                seg = quoted_segments[0]
                clip_input = {
                    "s3_uri": seg.get("s3_uri", ""),
                    "start_time": seg.get("start_time", 0),
                    "end_time": seg.get("end_time", 0)
                }
                clip_result = self._exec_create_subclip(clip_input)
                tool_calls.append({
                    "name": "create_subclip",
                    "input": clip_input,
                    "output_summary": self._summarize_result(clip_result)
                })
            else:
                clips = [
                    {"s3_uri": s.get("s3_uri", ""), "start_time": s.get("start_time", 0), "end_time": s.get("end_time", 0)}
                    for s in quoted_segments
                ]
                clip_input = {"clips": clips}
                clip_result = self._exec_concatenate_clips(clip_input)
                tool_calls.append({
                    "name": "concatenate_clips",
                    "input": clip_input,
                    "output_summary": self._summarize_result(clip_result)
                })

            if clip_result.get("type") not in ("error",):
                actions.append(clip_result)
                clip_s3_uri = clip_result.get("s3_uri")
            else:
                logger.warning(f"Clip creation failed: {clip_result.get('message')}")
        except Exception as e:
            logger.warning(f"Clip creation error (non-fatal): {e}")

        # Step 2: Analyze with Pegasus
        if clip_s3_uri:
            # Analyze the created clip — no time range needed since clip IS the segment
            analyze_input = {"s3_uri": clip_s3_uri, "prompt": message}
            logger.info(f"Quoted pipeline: analyzing clip {clip_s3_uri}")
            try:
                analysis_result = self._exec_analyze_video(analyze_input)
                actions.append(analysis_result)
                tool_calls.append({
                    "name": "analyze_video",
                    "input": analyze_input,
                    "output_summary": self._summarize_result(analysis_result)
                })
            except Exception as e:
                error_msg = f"Pegasus analysis failed (s3_uri={clip_s3_uri}): {str(e)}"
                logger.error(error_msg)
                actions.append({"type": "error", "message": error_msg})
                tool_calls.append({
                    "name": "analyze_video",
                    "input": analyze_input,
                    "output_summary": f"ERROR: {error_msg}"
                })
        else:
            # Fallback: analyze original video with time ranges
            video_segments = {}
            for seg in quoted_segments:
                vid_uri = seg.get("s3_uri", "")
                if vid_uri not in video_segments:
                    video_segments[vid_uri] = []
                video_segments[vid_uri].append(seg)

            for s3_uri, segs in video_segments.items():
                if len(segs) == 1:
                    seg = segs[0]
                    analyze_input = {
                        "s3_uri": s3_uri,
                        "prompt": message,
                        "start_time": seg.get("start_time", 0),
                        "end_time": seg.get("end_time", 0)
                    }
                else:
                    ranges = ", ".join(
                        f"{s.get('start_time', 0):.1f}s-{s.get('end_time', 0):.1f}s"
                        for s in segs
                    )
                    analyze_input = {
                        "s3_uri": s3_uri,
                        "prompt": f"Focus on these segments: {ranges}. {message}"
                    }

                logger.info(f"Quoted pipeline fallback: analyze_video(s3_uri={s3_uri})")
                try:
                    analysis_result = self._exec_analyze_video(analyze_input)
                    actions.append(analysis_result)
                    tool_calls.append({
                        "name": "analyze_video",
                        "input": analyze_input,
                        "output_summary": self._summarize_result(analysis_result)
                    })
                except Exception as e:
                    error_msg = f"Pegasus analysis failed (s3_uri={s3_uri}): {str(e)}"
                    logger.error(error_msg)
                    actions.append({"type": "error", "message": error_msg})
                    tool_calls.append({
                        "name": "analyze_video",
                        "input": analyze_input,
                        "output_summary": f"ERROR: {error_msg}"
                    })

        costs = self._calculate_costs(tool_calls)
        return {"message": "", "actions": actions, "tool_calls": tool_calls, "costs": costs}

    # ── System Prompt ──

    def _build_system_prompt(self, context: dict) -> str:
        selected_video = context.get("selected_video")
        quoted_segment = context.get("quoted_segment")
        quoted_segments = context.get("quoted_segments", [])
        if quoted_segment and not quoted_segments:
            quoted_segments = [quoted_segment]

        parts = [
            "You are a video analysis assistant with access to search and analysis tools.",
            "",
            "WHEN TO USE TOOLS:",
            "- Call search_segments or search_assets ONLY when the user explicitly asks to FIND or SEARCH for something new.",
            "- Call analyze_video ONLY when the user asks about the content of a specific video or segment.",
            "- Call create_subclip or concatenate_clips ONLY when the user has selected/quoted segments.",
            "",
            "WHEN NOT TO USE TOOLS:",
            "- Follow-up questions about results already returned — just answer from the tool results in conversation history.",
            "- Conversational messages (greetings, clarifications, opinions, comparisons of previous results).",
            "- If the user asks to elaborate on or summarize previous results — use the data already in the conversation.",
            "- DO NOT re-run a search just because the user sent a follow-up message.",
            "",
            "NEVER fabricate video content from your own knowledge. If you don't have tool results to answer from, say so.",
        ]

        # Quoted segments get top priority — must be before general rules
        if quoted_segments:
            parts.append("")
            parts.append("=" * 50)
            parts.append(f"QUOTED SEGMENTS ({len(quoted_segments)}) — THE USER HAS SELECTED THESE CLIPS:")
            for i, seg in enumerate(quoted_segments):
                parts.append(
                    f"  [{i+1}] Segment {seg.get('segment_id', 'N/A')} | "
                    f"{seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s | "
                    f"S3: {seg.get('s3_uri', 'N/A')}"
                )
            parts.append("")
            parts.append("MANDATORY INSTRUCTION FOR QUOTED SEGMENTS:")
            parts.append("DO NOT call search_segments or search_assets. The user already has clips.")
            parts.append("")
            if len(quoted_segments) == 1:
                seg = quoted_segments[0]
                parts.append(f"Step 1: Call create_subclip with s3_uri=\"{seg.get('s3_uri', '')}\", "
                             f"start_time={seg.get('start_time', 0)}, end_time={seg.get('end_time', 0)}")
                parts.append("")
                parts.append("Step 2 (AFTER create_subclip returns): You MUST call analyze_video.")
                parts.append("  - Use the s3_uri from the create_subclip result (the NEW s3_uri it returned, NOT the original).")
                parts.append("  - Use the user's question/request as the prompt.")
                parts.append("  - Do NOT skip this step. Do NOT answer without calling analyze_video.")
            else:
                parts.append("Step 1: Call concatenate_clips with clips=[")
                for seg in quoted_segments:
                    parts.append(f"  {{s3_uri: \"{seg.get('s3_uri', '')}\", "
                                 f"start_time: {seg.get('start_time', 0)}, end_time: {seg.get('end_time', 0)}}},")
                parts.append("]")
                parts.append("")
                parts.append("Step 2 (AFTER concatenate_clips returns): You MUST call analyze_video.")
                parts.append("  - Use the s3_uri from the concatenate_clips result (the NEW s3_uri it returned).")
                parts.append("  - Use the user's question/request as the prompt.")
                parts.append("  - Do NOT skip this step. Do NOT answer without calling analyze_video.")
            parts.append("")
            parts.append("BOTH steps are REQUIRED. You must call TWO tools in sequence.")
            parts.append("=" * 50)

        parts.extend([
            "",
            "TOOLS:",
            "- search_segments: Find clips/scenes/moments by semantic search. NEVER use when segments are quoted.",
            "- search_assets: Find full videos (aggregates by video). NEVER use when segments are quoted.",
            "- analyze_video: Answer questions about video content using Pegasus. Requires s3_uri.",
            "- create_subclip: Extract a time range from a video. Use BEFORE analyze_video on a single quoted segment.",
            "- concatenate_clips: Merge multiple subclips into one. Use BEFORE analyze_video on multiple quoted segments.",
            "",
            "RULES:",
            "1. Extract a CLEAN search query. Remove 'find me', 'show me', etc.",
            "2. 'clips'/'scenes'/'segments'/'moments' -> search_segments (only when NO segments are quoted)",
            "3. 'videos'/'assets'/'episodes' -> search_assets (only when NO segments are quoted)",
            "4. Questions about content ('describe', 'what happens', 'tell me', 'who', 'summary') -> analyze_video",
            "5. When video is selected (no quote) and user asks about content: analyze_video with the video's s3_uri.",
            "6. Extract limit from query ('find 5 clips' -> limit=5). Default: 10 segments, 5 assets.",
            "7. After tool results, give a brief summary based ONLY on tool output. Never fabricate.",
            "8. Pegasus constraint: video must be under 1 hour.",
        ])

        selected_videos = context.get("selected_videos", [])
        if selected_videos:
            parts.append("")
            parts.append("=" * 50)
            parts.append(f"SELECTED VIDEOS ({len(selected_videos)}):")
            for i, vid in enumerate(selected_videos):
                parts.append(f"  [{i+1}] {vid.get('name', 'Unknown')} | ID: {vid.get('video_id', 'N/A')} | S3: {vid.get('s3_uri', 'N/A')}")
            parts.append("")
            parts.append("When user asks about 'these videos', 'summarize', 'describe', 'what happens':")
            parts.append("  → Call analyze_video for each selected video using its S3 URI above.")
            parts.append("  → DO NOT say you lack information. The S3 URIs are provided above.")
            parts.append("=" * 50)
        elif selected_video:
            vid = selected_video
            parts.append("")
            parts.append("=" * 50)
            parts.append(f"SELECTED VIDEO: {vid.get('name', 'Unknown')}")
            parts.append(f"  Video ID: {vid.get('video_id', 'N/A')}")
            parts.append(f"  S3 URI: {vid.get('s3_uri', 'N/A')}")
            parts.append("")
            parts.append("When user asks 'summarize', 'describe', 'what happens', or any question about content:")
            parts.append(f"  → Call analyze_video with s3_uri=\"{vid.get('s3_uri', '')}\" and the user's question as prompt.")
            parts.append("  → DO NOT say you lack information. The S3 URI is provided above.")
            parts.append("=" * 50)

        return "\n".join(parts)

    # ── Message Builder ──

    def _build_messages(self, chat_history: list, current_message: str) -> list:
        messages = []
        for msg in chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": [{"text": content}]})
        messages.append({"role": "user", "content": [{"text": current_message}]})
        return messages

    # ── Tool Execution ──

    def _execute_tool(self, tool_name: str, tool_input: dict, settings: dict) -> dict:
        if tool_name == "search_segments":
            return self._exec_search_segments(tool_input, settings)
        elif tool_name == "search_assets":
            return self._exec_search_assets(tool_input, settings)
        elif tool_name == "analyze_video":
            return self._exec_analyze_video(tool_input)
        elif tool_name == "create_subclip":
            return self._exec_create_subclip(tool_input)
        elif tool_name == "concatenate_clips":
            return self._exec_concatenate_clips(tool_input)
        return {"type": "error", "message": f"Unknown tool: {tool_name}"}

    def _exec_search_segments(self, tool_input: dict, settings: dict) -> dict:
        query = tool_input["query"]
        limit = tool_input.get("limit", 10)
        video_id = tool_input.get("video_id")

        fusion_method = settings.get("fusion_method", "dynamic")
        backend = settings.get("backend", "s3vectors")
        use_multi_index = settings.get("use_multi_index", True)
        use_decomposition = settings.get("use_decomposition", False)
        query_image = settings.get("query_image")

        decomposed_queries = None
        if use_decomposition and query:
            decomposed_queries = self.search_client.bedrock.decompose_query(query)

        if fusion_method == "dynamic":
            response = self.search_client.search_dynamic(
                query=query,
                limit=limit,
                video_id=video_id,
                backend=backend,
                use_multi_index=use_multi_index,
                decomposed_queries=decomposed_queries,
                query_image=query_image
            )
            results = response["results"]
        else:
            modalities = None
            if fusion_method in ("visual", "audio", "transcription"):
                modalities = [fusion_method]
                fm = "weighted"
            else:
                fm = fusion_method if fusion_method in ("rrf", "weighted") else "weighted"

            results = self.search_client.search(
                query=query,
                modalities=modalities,
                limit=limit,
                video_id=video_id,
                fusion_method=fm,
                backend=backend,
                use_multi_index=use_multi_index,
                decomposed_queries=decomposed_queries,
                query_image=query_image
            )

        self._add_video_urls(results)
        return {"type": "search_results", "results": results}

    @staticmethod
    def _video_name_from_uri(s3_uri: str) -> str:
        """Extract human-readable name from proxy filename."""
        filename = os.path.basename(urlparse(s3_uri).path)
        name_no_ext = os.path.splitext(filename)[0]
        return name_no_ext.replace("_", " ").replace("-", " ") if name_no_ext else "Video"

    def _exec_search_assets(self, tool_input: dict, settings: dict) -> dict:
        query = tool_input["query"]
        limit = tool_input.get("limit", 5)

        # Over-fetch segments then aggregate by video
        segment_result = self._exec_search_segments(
            {"query": query, "limit": min(limit * 20, 100)},
            settings
        )
        segments = segment_result.get("results", [])

        video_map = {}
        for seg in segments:
            vid = seg.get("video_id", "")
            score = seg.get("confidence_score", seg.get("fusion_score", 0))
            if vid not in video_map:
                video_map[vid] = {
                    "video_id": vid,
                    "s3_uri": seg.get("s3_uri", ""),
                    "video_url": seg.get("video_url", ""),
                    "name": self._video_name_from_uri(seg.get("s3_uri", "")),
                    "best_segment": None,
                    "best_score": 0,
                    "segments": [],
                    "segment_count": 0
                }
            entry = video_map[vid]
            entry["segment_count"] += 1
            entry["segments"].append({
                "segment_id": seg.get("segment_id", 0),
                "start_time": seg.get("start_time", 0),
                "end_time": seg.get("end_time", 0),
                "score": score,
                "video_url": seg.get("video_url", ""),
                "modality_scores": seg.get("modality_scores", {}),
            })
            if score > entry["best_score"]:
                entry["best_score"] = score
                entry["best_segment"] = entry["segments"][-1]

        # Sort segments within each video by score desc, compute duration
        for v in video_map.values():
            v["segments"].sort(key=lambda s: s["score"], reverse=True)
            if v["segments"]:
                v["duration"] = max(s.get("end_time", 0) for s in v["segments"])

        ranked = sorted(video_map.values(), key=lambda x: x["best_score"], reverse=True)[:limit]
        return {"type": "asset_results", "results": ranked}

    def _exec_analyze_video(self, tool_input: dict) -> dict:
        s3_uri = tool_input["s3_uri"]
        prompt = tool_input["prompt"]
        start_time = tool_input.get("start_time")
        end_time = tool_input.get("end_time")

        logger.info(f"Pegasus analyze_video: s3_uri={s3_uri}, start={start_time}, end={end_time}")
        response_text = self.search_client.bedrock.analyze_video(
            s3_uri=s3_uri,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time
        )
        return {"type": "analysis_text", "text": response_text}

    def _exec_create_subclip(self, tool_input: dict) -> dict:
        payload = {
            "operation": "create_subclip",
            "s3_uri": tool_input["s3_uri"],
            "start_time": tool_input["start_time"],
            "end_time": tool_input["end_time"]
        }
        result = self._invoke_subclip_lambda(payload)
        if "error" in result:
            return {"type": "error", "message": result["error"]}
        s3_uri = result["s3_uri"]
        video_url = self._s3_uri_to_cloudfront(s3_uri)
        return {
            "type": "subclip",
            "s3_uri": s3_uri,
            "video_url": video_url,
            "duration": result["duration"]
        }

    def _exec_concatenate_clips(self, tool_input: dict) -> dict:
        payload = {
            "operation": "concatenate",
            "clips": tool_input["clips"]
        }
        result = self._invoke_subclip_lambda(payload)
        if "error" in result:
            return {"type": "error", "message": result["error"]}
        s3_uri = result["s3_uri"]
        video_url = self._s3_uri_to_cloudfront(s3_uri)
        return {
            "type": "concatenation",
            "s3_uri": s3_uri,
            "video_url": video_url,
            "duration": result["duration"],
            "clip_count": result.get("clip_count", len(tool_input["clips"]))
        }

    def _invoke_subclip_lambda(self, payload: dict) -> dict:
        logger.info(f"Invoking {SUBCLIP_LAMBDA_NAME}: {json.dumps(payload, default=str)}")
        response = self.lambda_client.invoke(
            FunctionName=SUBCLIP_LAMBDA_NAME,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )
        response_payload = json.loads(response["Payload"].read())
        if response.get("FunctionError"):
            logger.error(f"Lambda error: {response_payload}")
            return {"error": f"Lambda execution failed: {response_payload}"}
        return response_payload

    # ── Helpers ──

    @staticmethod
    def _normalize_s3_uri(s3_uri: str) -> str:
        """Normalize old S3 URIs to current bucket/path."""
        if s3_uri.startswith(OLD_S3_PREFIX):
            filename = s3_uri[len(OLD_S3_PREFIX):]
            return NEW_S3_PREFIX + filename
        return s3_uri

    def _s3_uri_to_cloudfront(self, s3_uri: str) -> str:
        """Convert an S3 URI to a CloudFront URL."""
        parsed = urlparse(s3_uri)
        key = parsed.path.lstrip("/")
        return f"https://{self.cloudfront_domain}/{key}"

    def _add_video_urls(self, results: list):
        for result in results:
            s3_uri = self._normalize_s3_uri(result.get("s3_uri", ""))
            result["s3_uri"] = s3_uri
            parsed = urlparse(s3_uri)
            key = parsed.path.lstrip("/")

            if "/proxies/" in key or key.startswith("proxies/"):
                proxy_key = key
            elif "WBD_project/Videos/proxy/" in key:
                proxy_key = "proxies/" + key.split("WBD_project/Videos/proxy/", 1)[1]
            elif key.startswith("input/"):
                proxy_key = key.replace("input/", "proxies/", 1)
            else:
                proxy_key = key

            result["video_url"] = f"https://{self.cloudfront_domain}/{proxy_key}"
            result["thumbnail_url"] = f"/api/thumbnail/{result.get('video_id', '')}/{result.get('segment_id', 0)}"

    def _summarize_result(self, result: dict) -> str:
        rtype = result.get("type", "unknown")
        if rtype == "search_results":
            return f"Found {len(result.get('results', []))} segments"
        elif rtype == "asset_results":
            return f"Found {len(result.get('results', []))} videos"
        elif rtype == "analysis_text":
            text = result.get("text", "")
            return text[:100] + "..." if len(text) > 100 else text
        elif rtype == "subclip":
            return f"Created subclip ({result.get('duration', 0):.1f}s) at {result.get('s3_uri', '')}"
        elif rtype == "concatenation":
            return f"Concatenated {result.get('clip_count', 0)} clips ({result.get('duration', 0):.1f}s)"
        elif rtype == "error":
            return result.get("message", "Error")
        return str(result)[:100]
