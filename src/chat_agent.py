"""
Chat Agent for the Analyze page.

Uses Bedrock Converse API with Claude to reason about user intent
and orchestrate tools: search (segments/assets), Pegasus analysis,
subclip creation, and concatenation via Lambda + FFmpeg.
"""

import json
import logging
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
        system_prompt = self._build_system_prompt(context)
        settings = context.get("settings", {})
        messages = self._build_messages(chat_history, message)

        actions = []
        tool_calls = []
        final_text = ""

        for turn in range(self.max_turns):
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

        return {"message": final_text, "actions": actions, "tool_calls": tool_calls}

    # ── System Prompt ──

    def _build_system_prompt(self, context: dict) -> str:
        selected_video = context.get("selected_video")
        quoted_segment = context.get("quoted_segment")

        parts = [
            "You are a video analysis assistant. You help users search through video content and analyze specific scenes.",
            "",
            "TOOLS:",
            "- search_segments: Find clips/scenes/moments by semantic search. Returns ranked segments with timestamps.",
            "- search_assets: Find full videos (aggregates by video). Use when user says 'videos', 'assets', or 'episodes'.",
            "- analyze_video: Answer questions about a video or segment using Pegasus video understanding.",
            "- create_subclip: Extract a time range from a video. Use before analyze_video on a quoted segment.",
            "- concatenate_clips: Merge multiple subclips into one video.",
            "",
            "RULES:",
            "1. Extract a CLEAN search query from natural language. Remove 'find me', 'show me', 'give me the best', etc.",
            "2. 'clips'/'scenes'/'segments'/'moments' -> search_segments",
            "3. 'videos'/'assets'/'episodes' -> search_assets",
            "4. Questions about content ('describe', 'what happens', 'who is speaking') -> analyze_video",
            "5. If a segment is quoted, use analyze_video with the quoted segment's s3_uri and time range.",
            "6. If a video is selected but no segment is quoted, and the user asks about content, use analyze_video on the full video.",
            "7. Extract result limit from query (e.g. 'find 5 clips' -> limit=5). Default: 10 segments, 5 assets.",
            "8. If no video is selected and user asks a general search, use search_segments.",
            "9. After tool results, give a brief helpful summary. Never fabricate results.",
            "10. Pegasus constraint: video must be under 1 hour.",
            "11. When a segment is quoted, first create_subclip to extract it, then analyze_video on the subclip S3 URI.",
            "12. When multiple segments are quoted, use concatenate_clips to merge them, then analyze_video on the result.",
        ]

        if selected_video:
            vid = selected_video
            parts.extend([
                "",
                f"SELECTED VIDEO: {vid.get('name', 'Unknown')}",
                f"  Video ID: {vid.get('video_id', 'N/A')}",
                f"  S3 URI: {vid.get('s3_uri', 'N/A')}",
            ])

        if quoted_segment:
            seg = quoted_segment
            parts.extend([
                "",
                f"QUOTED SEGMENT (user wants to analyze this):",
                f"  Video ID: {seg.get('video_id', 'N/A')}",
                f"  Segment ID: {seg.get('segment_id', 'N/A')}",
                f"  Time: {seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s",
                f"  S3 URI: {seg.get('s3_uri', 'N/A')}",
            ])

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
                decomposed_queries=decomposed_queries
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
                decomposed_queries=decomposed_queries
            )

        self._add_video_urls(results)
        return {"type": "search_results", "results": results}

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
            if vid not in video_map or score > video_map[vid]["best_score"]:
                video_map[vid] = {
                    "video_id": vid,
                    "s3_uri": seg.get("s3_uri", ""),
                    "video_url": seg.get("video_url", ""),
                    "best_segment": {
                        "segment_id": seg.get("segment_id", 0),
                        "start_time": seg.get("start_time", 0),
                        "end_time": seg.get("end_time", 0),
                        "score": score
                    },
                    "best_score": score,
                    "segment_count": 0
                }
            video_map[vid]["segment_count"] += 1

        ranked = sorted(video_map.values(), key=lambda x: x["best_score"], reverse=True)[:limit]
        return {"type": "asset_results", "results": ranked}

    def _exec_analyze_video(self, tool_input: dict) -> dict:
        s3_uri = tool_input["s3_uri"]
        prompt = tool_input["prompt"]
        start_time = tool_input.get("start_time")
        end_time = tool_input.get("end_time")

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
        return {
            "type": "subclip",
            "s3_uri": result["s3_uri"],
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
        return {
            "type": "concatenation",
            "s3_uri": result["s3_uri"],
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

    def _add_video_urls(self, results: list):
        for result in results:
            s3_uri = result.get("s3_uri", "")
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
