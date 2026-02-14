"""
Director Agent — coordinates song creation via subagents.

Phase 1 (Planning):  Interpret instructions, plan song structure, delegate instruments.
Phase 2 (Arrangement): Position layers, validate gaps, finalise.

The director never writes beats or notes itself — subagents do that.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.models.schemas import (
    AudioSegment, SegmentType, MusicDescription,
    Layer, Project, Genre,
)
from app.services.subagents import DrumsSubagent, MelodySubagent, VocalSubagent
from app.services.subagents.base import audio_duration, _get_client

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 tools — planning & delegation
# ═══════════════════════════════════════════════════════════════════════════

PHASE1_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan_song",
            "description": (
                "Set the song's tempo, genre, key, total bars, and define sections. "
                "Call this FIRST before delegating instruments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bpm": {
                        "type": "integer",
                        "description": "Tempo in BPM (40–200)",
                    },
                    "genre": {
                        "type": "string",
                        "enum": ["hip-hop", "pop", "rock", "jazz", "electronic", "lo-fi", "rnb"],
                    },
                    "key": {
                        "type": "string",
                        "description": "Key signature, e.g. 'C major', 'A minor'",
                    },
                    "total_bars": {
                        "type": "integer",
                        "description": "Total song length in bars (e.g. 16, 32)",
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "e.g. intro, verse, chorus, bridge, outro"},
                                "start_bar": {"type": "integer"},
                                "end_bar": {"type": "integer"},
                                "description": {"type": "string", "description": "Brief style note for this section"},
                            },
                            "required": ["name", "start_bar", "end_bar"],
                        },
                    },
                },
                "required": ["bpm", "genre", "total_bars", "sections"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_drums",
            "description": (
                "Assign percussion to sections. The drums subagent will create the patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "start_bar": {"type": "integer"},
                                "end_bar": {"type": "integer"},
                                "description": {"type": "string"},
                            },
                            "required": ["name", "start_bar", "end_bar"],
                        },
                        "description": "Sections the drums should cover",
                    },
                    "style_notes": {
                        "type": "string",
                        "description": "Style guidance for the drummer (e.g. 'heavy kick, trap hi-hats')",
                    },
                    "segment_id": {
                        "type": "string",
                        "description": "Optional: user's beatboxing segment ID to render",
                    },
                },
                "required": ["sections"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_melody",
            "description": (
                "Assign a melodic instrument + role to sections. "
                "Call multiple times for different instruments (e.g. piano chords + bass line)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instrument": {
                        "type": "string",
                        "enum": ["piano", "guitar", "bass", "synth", "strings"],
                    },
                    "role": {
                        "type": "string",
                        "enum": ["melody", "chords", "bass_line", "pad"],
                        "description": "What this instrument should play",
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "start_bar": {"type": "integer"},
                                "end_bar": {"type": "integer"},
                                "description": {"type": "string"},
                            },
                            "required": ["name", "start_bar", "end_bar"],
                        },
                    },
                    "style_notes": {
                        "type": "string",
                        "description": "Style guidance (e.g. 'jazzy chord voicings', 'walking bass')",
                    },
                    "segment_id": {
                        "type": "string",
                        "description": "Optional: user's hummed melody segment ID to render",
                    },
                },
                "required": ["instrument", "role", "sections"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_vocal",
            "description": (
                "Create vocals via TTS synthesis. Provide either a speech segment_id "
                "(lyrics extracted from transcript) or explicit lyrics text. "
                "The vocal subagent designs a melody or rhythm, then TTS generates clean audio per word."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_id": {
                        "type": "string",
                        "description": "Optional: speech segment ID to extract lyrics from its transcript",
                    },
                    "lyrics": {
                        "type": "string",
                        "description": "Optional: explicit lyrics text (alternative to segment_id)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["melodic", "rhythmic"],
                        "description": "Vocal style: 'melodic' for singing (pop/R&B/jazz), 'rhythmic' for rap/spoken word (hip-hop/electronic)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Layer name (e.g. 'Lead Vocals', 'Rap Flow')",
                    },
                },
                "required": ["mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_planning",
            "description": "Signal that all delegations are done. Subagents will now execute.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 tools — arrangement & completion
# ═══════════════════════════════════════════════════════════════════════════

PHASE2_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "arrange_layer",
            "description": (
                "Position a layer at a specific bar. The layer plays from start_bar "
                "for its audio duration, then STOPS. It does NOT loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {"type": "string", "description": "Layer ID"},
                    "start_bar": {"type": "integer", "description": "Bar to start at (0-indexed)"},
                },
                "required": ["layer_id", "start_bar"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_layer",
            "description": "Remove a layer from the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "layer_id": {"type": "string", "description": "Layer ID or name"},
                },
                "required": ["layer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete",
            "description": (
                "Finalise the arrangement. Will REJECT if silent gaps > 2 bars exist. "
                "Fix gaps first, then call again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Brief summary of what was created"},
                },
                "required": ["summary"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Director
# ═══════════════════════════════════════════════════════════════════════════

class Director:
    """Two-phase director: plan + delegate, then arrange + validate."""

    max_phase1_iterations = 8
    max_phase2_iterations = 6

    async def run(
        self,
        project: Project,
        segments: list[AudioSegment],
        speech_instructions: list[str],
        segment_data: dict[str, dict],
        segment_audio_data: dict[str, bytes],
    ) -> str:
        """
        Full orchestration: plan, delegate to subagents, arrange.
        Returns a summary string.
        """
        logger.info("=" * 60)
        logger.info("DIRECTOR: Starting orchestration")
        self._current_segments = segments

        # ── Phase 1: Planning ──────────────────────────────────────────
        context = self._build_context(segments, speech_instructions, segment_data)
        delegations, song_plan = await self._phase1(project, context, segment_data, segment_audio_data)

        if not delegations:
            logger.warning("DIRECTOR: No delegations produced — falling back to default")
            delegations = self._default_delegations(project, segment_data, segment_audio_data)

        # ── Execute subagents concurrently ─────────────────────────────
        all_layers = await self._run_subagents(project, delegations, segment_data, segment_audio_data)

        # Add layers to project
        project.layers.extend(all_layers)
        logger.info(f"DIRECTOR: Subagents produced {len(all_layers)} layers total")

        # ── Phase 2: Arrangement ───────────────────────────────────────
        summary = await self._phase2(project, song_plan)

        logger.info(f"DIRECTOR: Done — {summary}")
        logger.info("=" * 60)
        return summary

    async def run_refinement(
        self,
        project: Project,
        instructions: str,
        segments: list[AudioSegment] | None,
        segment_data: dict[str, dict],
        segment_audio_data: dict[str, bytes],
    ) -> str:
        """
        Refinement mode: selective re-delegation or direct arrangement changes.
        """
        logger.info("=" * 60)
        logger.info(f"DIRECTOR: Starting refinement — {instructions}")
        self._current_segments = segments or []

        context = self._build_refinement_context(project, instructions, segments, segment_data)
        delegations, _ = await self._phase1_refinement(
            project, context, instructions, segment_data, segment_audio_data
        )

        if delegations:
            new_layers = await self._run_subagents(project, delegations, segment_data, segment_audio_data)
            project.layers.extend(new_layers)
            logger.info(f"DIRECTOR: Refinement subagents produced {len(new_layers)} layers")

        summary = await self._phase2(project, {})
        logger.info(f"DIRECTOR: Refinement done — {summary}")
        logger.info("=" * 60)
        return summary

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Planning
    # ═══════════════════════════════════════════════════════════════════

    async def _phase1(
        self,
        project: Project,
        context: str,
        segment_data: dict,
        segment_audio: dict,
    ) -> tuple[list[dict], dict]:
        """Run the planning LLM loop. Returns (delegations, song_plan)."""
        messages = [
            {"role": "system", "content": self._phase1_system_prompt()},
            {"role": "user", "content": context},
        ]

        delegations: list[dict] = []
        song_plan: dict = {}

        for iteration in range(self.max_phase1_iterations):
            logger.info(f"DIRECTOR P1: Iteration {iteration + 1}")
            response = await _get_client().chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=PHASE1_TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            messages.append(msg)

            if msg.content:
                logger.info(f"DIRECTOR P1: {msg.content}")

            if not msg.tool_calls:
                break

            finish_signaled = False
            for tc in msg.tool_calls:
                fname = tc.function.name
                args = json.loads(tc.function.arguments)
                logger.info(f"DIRECTOR P1: {fname}({json.dumps(args, indent=None)[:200]})")

                result = self._handle_phase1_tool(
                    fname, args, project, delegations, song_plan,
                    segment_data, segment_audio,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

                if fname == "finish_planning":
                    finish_signaled = True

            if finish_signaled:
                self._auto_delegate_vocals(delegations, song_plan)
                logger.info(f"DIRECTOR P1: Planning complete — {len(delegations)} delegations")
                return delegations, song_plan

        self._auto_delegate_vocals(delegations, song_plan)
        return delegations, song_plan

    def _handle_phase1_tool(
        self, name: str, args: dict, project: Project,
        delegations: list[dict], song_plan: dict,
        segment_data: dict, segment_audio: dict,
    ) -> dict:
        if name == "plan_song":
            bpm = max(40, min(200, args.get("bpm", 120)))
            project.bpm = bpm
            genre_str = args.get("genre", "pop")
            try:
                genre = Genre(genre_str)
                if project.description is None:
                    project.description = MusicDescription(genre=genre, instruments=[])
                else:
                    project.description.genre = genre
            except ValueError:
                pass
            if args.get("key"):
                project.key_signature = args["key"]

            song_plan["total_bars"] = args.get("total_bars", 16)
            song_plan["sections"] = args.get("sections", [])
            song_plan["genre"] = genre_str

            return {
                "success": True,
                "message": f"Song plan set: {bpm} BPM, {genre_str}, {song_plan['total_bars']} bars, "
                           f"{len(song_plan['sections'])} sections",
            }

        elif name == "delegate_drums":
            d = {
                "type": "drums",
                "sections": args.get("sections", []),
                "style_notes": args.get("style_notes", ""),
                "genre": song_plan.get("genre", "pop"),
                "segment_id": args.get("segment_id"),
                "segment_data": segment_data,
                "segment_audio": segment_audio,
            }
            delegations.append(d)
            return {"success": True, "message": f"Drums delegated for {len(d['sections'])} sections"}

        elif name == "delegate_melody":
            d = {
                "type": "melody",
                "instrument": args["instrument"],
                "role": args["role"],
                "sections": args.get("sections", []),
                "style_notes": args.get("style_notes", ""),
                "genre": song_plan.get("genre", "pop"),
                "segment_id": args.get("segment_id"),
                "segment_data": segment_data,
                "segment_audio": segment_audio,
            }
            delegations.append(d)
            return {
                "success": True,
                "message": f"Delegated {args['instrument']} ({args['role']}) for {len(d['sections'])} sections",
            }

        elif name == "delegate_vocal":
            sid = args.get("segment_id")
            lyrics = args.get("lyrics")
            mode = args.get("mode", "melodic")

            words = []
            transcript = ""

            if sid:
                # Resolve segment ID — find the AudioSegment for word timestamps
                resolved_sid = None
                if sid in segment_audio:
                    resolved_sid = sid
                else:
                    for full_id in segment_audio:
                        if full_id.startswith(sid) or sid.startswith(full_id):
                            resolved_sid = full_id
                            break

                if resolved_sid is None:
                    available = list(segment_audio.keys())
                    if available:
                        short_ids = [s[:8] for s in available]
                        return {
                            "success": False,
                            "error": f"Segment '{sid}' not found. Available: {short_ids}",
                        }

                # Get words and transcript from the segment
                for seg in self._current_segments:
                    if seg.id == resolved_sid or seg.id.startswith(sid) or sid.startswith(seg.id):
                        words = seg.words or []
                        transcript = seg.transcript or " ".join(w["word"] for w in words)
                        break

            elif lyrics:
                # Build word list from explicit lyrics text
                transcript = lyrics
                words = [{"word": w} for w in lyrics.split() if w.strip()]

            else:
                # Auto-pick first speech segment with words
                for seg in self._current_segments:
                    if seg.type == SegmentType.SPEECH and seg.words and len(seg.words) > 0:
                        words = seg.words
                        transcript = seg.transcript or " ".join(w["word"] for w in words)
                        logger.info(f"DIRECTOR P1: delegate_vocal auto-picked speech segment {seg.id[:8]}")
                        break

                if not words:
                    return {
                        "success": False,
                        "error": "No speech segments or lyrics available for vocals.",
                    }

            if not words:
                return {
                    "success": False,
                    "error": "No words found — segment has no transcript or lyrics are empty.",
                }

            d = {
                "type": "vocal",
                "mode": mode,
                "name": args.get("name", "Vocals"),
                "words": words,
                "transcript": transcript,
                "genre": song_plan.get("genre", "pop"),
                "style_notes": args.get("style_notes", ""),
                "sections": song_plan.get("sections", []),
            }
            delegations.append(d)
            return {
                "success": True,
                "message": f"Vocal delegated: {mode} mode, {len(words)} words",
            }

        elif name == "finish_planning":
            return {"success": True, "message": "Planning complete — executing subagents now"}

        return {"success": False, "error": f"Unknown tool: {name}"}

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1 Refinement
    # ═══════════════════════════════════════════════════════════════════

    async def _phase1_refinement(
        self,
        project: Project,
        context: str,
        instructions: str,
        segment_data: dict,
        segment_audio: dict,
    ) -> tuple[list[dict], dict]:
        """Refinement planning — decides which subagents to re-invoke."""
        messages = [
            {"role": "system", "content": self._phase1_refinement_system_prompt()},
            {"role": "user", "content": context},
        ]

        delegations: list[dict] = []
        song_plan: dict = {}

        for iteration in range(self.max_phase1_iterations):
            logger.info(f"DIRECTOR REFINE P1: Iteration {iteration + 1}")
            response = await _get_client().chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=PHASE1_TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            messages.append(msg)

            if msg.content:
                logger.info(f"DIRECTOR REFINE P1: {msg.content}")

            if not msg.tool_calls:
                break

            finish_signaled = False
            for tc in msg.tool_calls:
                fname = tc.function.name
                args = json.loads(tc.function.arguments)
                logger.info(f"DIRECTOR REFINE P1: {fname}")

                result = self._handle_phase1_tool(
                    fname, args, project, delegations, song_plan,
                    segment_data, segment_audio,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

                if fname == "finish_planning":
                    finish_signaled = True

            if finish_signaled:
                return delegations, song_plan

        return delegations, song_plan

    # ═══════════════════════════════════════════════════════════════════
    # Subagent execution
    # ═══════════════════════════════════════════════════════════════════

    async def _run_subagents(
        self,
        project: Project,
        delegations: list[dict],
        segment_data: dict,
        segment_audio: dict,
    ) -> list[Layer]:
        """Run all delegated subagents concurrently. Returns combined layers."""
        tasks = []
        for d in delegations:
            if d["type"] == "drums":
                agent = DrumsSubagent()
                tasks.append(agent.run(project, d))
            elif d["type"] == "melody":
                agent = MelodySubagent(instrument=d["instrument"], role=d["role"])
                tasks.append(agent.run(project, d))
            elif d["type"] == "vocal":
                agent = VocalSubagent(mode=d.get("mode", "melodic"))
                tasks.append(agent.run(project, d))

        if not tasks:
            return []

        logger.info(f"DIRECTOR: Launching {len(tasks)} subagents concurrently")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_layers: list[Layer] = []
        for i, result in enumerate(results):
            d_type = delegations[i]["type"] if i < len(delegations) else "unknown"
            if isinstance(result, Exception):
                logger.error(f"DIRECTOR: Subagent {i} ({d_type}) failed: {result}")
            elif isinstance(result, list):
                if len(result) == 0:
                    logger.warning(
                        f"DIRECTOR: Subagent {i} ({d_type}) produced 0 layers — "
                        f"delegation may have had invalid parameters"
                    )
                else:
                    all_layers.extend(result)
                    logger.info(f"DIRECTOR: Subagent {i} ({d_type}) produced {len(result)} layers")

        return all_layers

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2: Arrangement
    # ═══════════════════════════════════════════════════════════════════

    async def _phase2(self, project: Project, song_plan: dict) -> str:
        """Run the arrangement LLM loop. Returns summary string."""
        if not project.layers:
            return "No layers to arrange"

        context = self._build_phase2_context(project, song_plan)
        messages = [
            {"role": "system", "content": self._phase2_system_prompt()},
            {"role": "user", "content": context},
        ]

        summary = "Track created successfully"
        completed = False

        for iteration in range(self.max_phase2_iterations):
            if completed:
                break

            logger.info(f"DIRECTOR P2: Iteration {iteration + 1}")
            response = await _get_client().chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=PHASE2_TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            messages.append(msg)

            if msg.content:
                logger.info(f"DIRECTOR P2: {msg.content}")

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                fname = tc.function.name
                args = json.loads(tc.function.arguments)
                logger.info(f"DIRECTOR P2: {fname}({json.dumps(args, indent=None)[:200]})")

                result = self._handle_phase2_tool(fname, args, project)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

                if fname == "complete" and result.get("success"):
                    summary = args.get("summary", summary)
                    completed = True

        return summary

    def _handle_phase2_tool(self, name: str, args: dict, project: Project) -> dict:
        if name == "arrange_layer":
            layer_id = args["layer_id"]
            start_bar = args["start_bar"]

            target = self._find_layer(project, layer_id)
            if target is None:
                return {"success": False, "error": f"Layer '{layer_id}' not found"}

            target.start_bar = max(0, start_bar)
            dur = audio_duration(target.audio_file) if target.audio_file else 0.0
            spb = 60.0 / project.bpm
            start_sec = target.start_bar * 4 * spb
            end_sec = start_sec + dur
            end_bar = target.start_bar + dur / (4 * spb) if spb > 0 else 0
            return {
                "success": True,
                "message": f"'{target.name}' at bar {target.start_bar} "
                           f"(plays {start_sec:.1f}s–{end_sec:.1f}s, ends bar {end_bar:.1f})",
            }

        elif name == "remove_layer":
            layer_id = args["layer_id"]
            for lyr in list(project.layers):
                if lyr.id == layer_id or lyr.id.startswith(layer_id) or lyr.name.lower() == layer_id.lower():
                    project.layers.remove(lyr)
                    return {"success": True, "message": f"Removed '{lyr.name}'"}
            return {"success": False, "error": f"Layer '{layer_id}' not found"}

        elif name == "complete":
            return self._check_gaps_and_complete(project)

        return {"success": False, "error": f"Unknown tool: {name}"}

    def _check_gaps_and_complete(self, project: Project) -> dict:
        """Validate no silent gaps > 2 bars, same as old agent's complete tool."""
        if not project.layers or not project.bpm:
            return {"success": True, "message": "Orchestration complete"}

        spb = 60.0 / project.bpm
        bar_sec = 4 * spb

        intervals = []
        for lyr in project.layers:
            if lyr.audio_file:
                dur = audio_duration(lyr.audio_file)
                start = lyr.start_bar * bar_sec
                intervals.append((start, start + dur))

        if not intervals:
            return {"success": True, "message": "Orchestration complete"}

        intervals.sort()
        covered_until = 0.0
        gaps = []
        for start, end in intervals:
            if start > covered_until + bar_sec * 2:
                gaps.append((covered_until, start))
            covered_until = max(covered_until, end)

        if gaps:
            gap_desc = "; ".join(
                f"{g[0]:.1f}s–{g[1]:.1f}s (bar {g[0]/bar_sec:.0f}–{g[1]/bar_sec:.0f})"
                for g in gaps
            )
            return {
                "success": False,
                "error": f"Silent gaps detected: {gap_desc}. "
                         f"Move layers to fill gaps, then call complete again.",
            }

        return {"success": True, "message": "Orchestration complete"}

    # ═══════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════

    def _auto_delegate_vocals(self, delegations: list[dict], song_plan: dict):
        """Auto-add a vocal delegation if the LLM didn't and speech segments exist."""
        has_vocal = any(d["type"] == "vocal" for d in delegations)
        if has_vocal:
            return

        for seg in self._current_segments:
            if seg.type == SegmentType.SPEECH and seg.words and len(seg.words) > 0:
                words = seg.words
                transcript = seg.transcript or " ".join(w["word"] for w in words)
                genre = song_plan.get("genre", "pop")
                mode = "rhythmic" if genre in ("hip-hop", "electronic") else "melodic"
                d = {
                    "type": "vocal",
                    "mode": mode,
                    "name": "Vocals",
                    "words": words,
                    "transcript": transcript,
                    "genre": genre,
                    "style_notes": "",
                    "sections": song_plan.get("sections", []),
                }
                delegations.append(d)
                logger.info(
                    f"DIRECTOR: Auto-delegated vocals ({mode} mode, "
                    f"{len(words)} words from segment {seg.id[:8]})"
                )
                return

    def _find_layer(self, project: Project, layer_id: str) -> Layer | None:
        for lyr in project.layers:
            if lyr.id == layer_id or lyr.id.startswith(layer_id):
                return lyr
        return None

    def _default_delegations(
        self, project: Project,
        segment_data: dict, segment_audio: dict,
    ) -> list[dict]:
        """Fallback when LLM produces no delegations — create a basic drum + piano."""
        total_bars = 8
        sections = [{"name": "main", "start_bar": 0, "end_bar": total_bars}]
        delegations = [
            {
                "type": "drums",
                "sections": sections,
                "style_notes": "basic pattern",
                "genre": "pop",
                "segment_data": segment_data,
                "segment_audio": segment_audio,
            },
            {
                "type": "melody",
                "instrument": "piano",
                "role": "chords",
                "sections": sections,
                "style_notes": "simple chords",
                "genre": "pop",
                "segment_data": segment_data,
                "segment_audio": segment_audio,
            },
        ]
        # Also delegate any user segments
        for sid, data in segment_data.items():
            if data["type"] == "rhythm":
                delegations[0]["segment_id"] = sid
            elif data["type"] == "melody":
                delegations[1]["segment_id"] = sid
        return delegations

    # ═══════════════════════════════════════════════════════════════════
    # System prompts
    # ═══════════════════════════════════════════════════════════════════

    def _phase1_system_prompt(self) -> str:
        return """You are a music director. You plan songs but do NOT create instrument parts.

Your job:
1. Interpret the user's instructions (speech transcripts + audio segments)
2. Call plan_song to set tempo, genre, key, total bars, and sections
3. Delegate instruments:
   - delegate_drums for percussion
   - delegate_melody for each melodic instrument (call multiple times for piano + bass + etc.)
   - delegate_vocal for vocals — extracts lyrics from speech transcripts and synthesizes via TTS
4. Call finish_planning when all delegations are done

## IMPORTANT: When to delegate vocals
- If the user's speech contains lyrics (even mixed with instructions), ALWAYS delegate vocals
- If the user mentions "vocals", "singing", "lyrics", "rap", "words", etc., ALWAYS delegate vocals
- The vocal subagent will filter out instruction words and keep only actual lyrics
- When in doubt, delegate vocals — the subagent handles lyrics extraction intelligently

## CRITICAL: Follow the user's instructions exactly
The user's requests take absolute priority over any default structure.
If they say "short", "quick", "simple", "just a loop", "4 bars", etc. — make it SHORT.
If they specify a bar count or duration, use exactly that.
Do NOT pad out to a longer structure than requested.

## Section design
Choose total_bars based on the user's request:
- Short/loop/simple: 4–8 bars (1–2 sections)
- Medium/default (no length preference stated): 8–16 bars
- Full song: 16–40 bars (intro/verse/chorus/etc.)
- Sections should not overlap, and should cover the full song

## Delegation tips
- ALWAYS delegate at least drums + one melodic instrument
- For a full sound: drums + bass + chords (piano/guitar) + optional melody/pad
- If the user provided a rhythm segment, pass its ID to delegate_drums
- If the user provided a melody segment, pass its ID to delegate_melody
- Match complexity to the user's request: short songs need fewer instruments

## Vocals (TTS-based pipeline)
- delegate_vocal synthesizes vocals via TTS — no raw audio segment needed
- Provide segment_id (extracts lyrics from speech transcript) OR lyrics (explicit text)
- mode='melodic' for singing (pop/R&B/jazz/lo-fi) — TTS words are pitch-shifted to MIDI notes
- mode='rhythmic' for rap/spoken word (hip-hop/electronic/rock) — TTS words are time-snapped to beat grid
- The vocal subagent filters out instruction words, keeping only actual lyrics
- If the user asks for vocals and has a speech segment, use its segment_id to extract lyrics"""

    def _phase1_refinement_system_prompt(self) -> str:
        return """You are a music director refining an existing track.

## CRITICAL: Follow the user's instructions exactly
If they say "shorter", "make it shorter", "cut it down", "fewer bars", etc.:
- Call plan_song with a reduced total_bars and fewer/shorter sections
- Re-delegate instruments with the shorter section assignments
- The arrangement phase will then position the new shorter layers

The user wants specific changes. You should:
1. Only delegate the instruments that need to change
2. Use plan_song if tempo/genre/structure/length changes are needed
3. Use delegate_drums / delegate_melody / delegate_vocal for new or replacement parts
4. Call finish_planning when done (even if no delegations — arrangement phase will handle removes/moves)

Be surgical: if the user says "change the drums", only delegate drums.
If they say "add bass", only delegate a bass melody subagent.
If they say "add vocals" or "add singing", delegate_vocal with lyrics or a speech segment_id.
If they say "make it shorter", call plan_song with fewer bars then re-delegate ALL instruments with shorter sections.
For simple operations (remove a layer, rearrange), just call finish_planning
with no delegations — the arrangement phase will handle it."""

    def _phase2_system_prompt(self) -> str:
        return """All instrument parts are ready. Your job is to arrange them into a complete song.

## What you can do
- arrange_layer: position a layer at a specific bar
- remove_layer: delete a layer
- complete: finalise (will reject if silent gaps > 2 bars exist)

## Arrangement rules
1. Each layer plays from its start_bar for its audio duration, then STOPS (no looping)
2. Position layers so they cover their intended sections
3. Avoid gaps > 2 bars between sounding content
4. Layers can overlap (that's fine — they mix together)
5. Check each layer's audio duration and bars to plan placement

Call complete with a brief summary when the arrangement is solid."""

    # ═══════════════════════════════════════════════════════════════════
    # Context builders
    # ═══════════════════════════════════════════════════════════════════

    def _build_context(
        self,
        segments: list[AudioSegment],
        speech_instructions: list[str],
        segment_data: dict[str, dict],
    ) -> str:
        """Build context for Phase 1 planning."""
        lines = ["# User Recording Analysis\n"]

        lines.append("## Speech Instructions (in order):")
        for i, text in enumerate(speech_instructions, 1):
            lines.append(f'{i}. "{text}"')
        if not speech_instructions:
            lines.append("(none — create something interesting)")
        lines.append("")

        lines.append("## Musical Segments:")
        has_musical = False
        for segment in segments:
            if segment.type == SegmentType.SPEECH:
                continue
            has_musical = True
            data = segment_data.get(segment.id, {})
            duration = segment.end_seconds - segment.start_seconds

            lines.append(f"\n### Segment {segment.id} ({segment.type.value})")
            lines.append(f"- Duration: {duration:.2f}s")

            if data.get("type") == "rhythm":
                rhythm = data.get("rhythm")
                if rhythm:
                    lines.append(f"- Detected BPM: {rhythm.bpm}")
                    lines.append(f"- Beats: {len(rhythm.beats)}, Bars: {rhythm.bars}")
            elif data.get("type") == "melody":
                melody = data.get("melody")
                if melody:
                    lines.append(f"- Notes: {len(melody.pitches)}")
                    if melody.key_signature:
                        lines.append(f"- Key: {melody.key_signature}")

        if not has_musical:
            lines.append("(none — create everything from scratch)")
        lines.append("")

        # List segments available for delegate_vocal
        vocal_candidates = []
        for segment in segments:
            if segment.type == SegmentType.SPEECH and segment.words and len(segment.words) > 2:
                word_count = len(segment.words)
                transcript = segment.transcript or ""
                if len(transcript) > 60:
                    vocal_candidates.append(
                        f"- {segment.id} (speech, {word_count} words: \"{transcript[:60]}...\")"
                    )
                else:
                    vocal_candidates.append(
                        f"- {segment.id} (speech, {word_count} words: \"{transcript}\")"
                    )
            elif segment.type == SegmentType.MELODY:
                vocal_candidates.append(f"- {segment.id} (singing/melody)")

        if vocal_candidates:
            lines.append("## Segments available for delegate_vocal (TTS-synthesized):")
            lines.append("Lyrics from speech transcripts will be synthesized via TTS (clean per-word audio).")
            lines.append("Use mode='melodic' for singing, mode='rhythmic' for rap/spoken word.")
            lines.extend(vocal_candidates)
            lines.append("")
        else:
            lines.append("## Vocals: No segments available for vocals.\n")
            lines.append("You can still use delegate_vocal with explicit lyrics text.")
            lines.append("")

        lines.append("## Your Task:")
        lines.append("Plan the song structure and delegate instruments to subagents.")
        return "\n".join(lines)

    def _build_refinement_context(
        self,
        project: Project,
        instructions: str,
        segments: list[AudioSegment] | None,
        segment_data: dict[str, dict],
    ) -> str:
        """Build context for refinement Phase 1."""
        lines = ["# Current Project\n"]

        genre_str = project.description.genre.value if project.description and project.description.genre else "not set"
        lines.append(f"- BPM: {project.bpm}, Key: {project.key_signature or 'not set'}, Genre: {genre_str}")
        lines.append("")

        lines.append("## Existing Layers:")
        if project.layers:
            for i, lyr in enumerate(project.layers, 1):
                inst = lyr.instrument.value if lyr.instrument else lyr.segment_type.value
                dur = audio_duration(lyr.audio_file) if lyr.audio_file else 0.0
                lines.append(f"{i}. [id:{lyr.id}] {lyr.name} — {inst}, bar {lyr.start_bar}, {dur:.1f}s audio")
        else:
            lines.append("(no layers yet)")
        lines.append("")

        lines.append(f'## User Request:\n"{instructions}"\n')

        if segments:
            lines.append("## New Audio Segments:")
            for seg in segments:
                if seg.type == SegmentType.SPEECH:
                    continue
                data = segment_data.get(seg.id, {})
                duration = seg.end_seconds - seg.start_seconds
                lines.append(f"- Segment {seg.id} ({seg.type.value}): {duration:.2f}s")
                if data.get("type") == "rhythm" and data.get("rhythm"):
                    lines.append(f"  BPM: {data['rhythm'].bpm}, Beats: {len(data['rhythm'].beats)}")
                elif data.get("type") == "melody" and data.get("melody"):
                    lines.append(f"  Notes: {len(data['melody'].pitches)}")
            lines.append("")

        # Vocal availability
        vocal_candidates = []
        if segments:
            for seg in segments:
                if seg.type == SegmentType.SPEECH and seg.words and len(seg.words) > 2:
                    word_count = len(seg.words)
                    transcript = seg.transcript or ""
                    if len(transcript) > 60:
                        vocal_candidates.append(
                            f"- {seg.id} (speech, {word_count} words: \"{transcript[:60]}...\")"
                        )
                    else:
                        vocal_candidates.append(
                            f"- {seg.id} (speech, {word_count} words: \"{transcript}\")"
                        )
                elif seg.type == SegmentType.MELODY:
                    vocal_candidates.append(f"- {seg.id} (singing/melody)")

        if vocal_candidates:
            lines.append("## Segments available for delegate_vocal (TTS-synthesized):")
            lines.append("Lyrics from speech transcripts will be synthesized via TTS (clean per-word audio).")
            lines.append("Use mode='melodic' for singing, mode='rhythmic' for rap/spoken word.")
            lines.extend(vocal_candidates)
            lines.append("")
        else:
            lines.append("## Vocals: No segments available for vocals.\n")
            lines.append("You can still use delegate_vocal with explicit lyrics text.")
            lines.append("")

        lines.append("## Your Task:")
        lines.append("Decide what needs to change. Delegate only the instruments that need work.")
        lines.append("For simple changes (remove/rearrange), just call finish_planning.")
        return "\n".join(lines)

    def _build_phase2_context(self, project: Project, song_plan: dict) -> str:
        """Build context for Phase 2 arrangement."""
        lines = ["# Layers Ready for Arrangement\n"]
        lines.append(f"BPM: {project.bpm}")

        spb = 60.0 / project.bpm
        bar_sec = 4 * spb

        lines.append("")
        for i, lyr in enumerate(project.layers, 1):
            inst = lyr.instrument.value if lyr.instrument else lyr.segment_type.value
            dur = audio_duration(lyr.audio_file) if lyr.audio_file else 0.0
            dur_bars = dur / bar_sec if bar_sec > 0 else 0
            lines.append(
                f"{i}. [id:{lyr.id}] {lyr.name} — {inst}, "
                f"currently at bar {lyr.start_bar}, "
                f"{dur:.1f}s audio ({dur_bars:.1f} bars)"
            )

        if song_plan.get("sections"):
            lines.append("\n## Planned Sections:")
            for sec in song_plan["sections"]:
                lines.append(f"- {sec['name']}: bars {sec['start_bar']}–{sec['end_bar']}")

        lines.append("\n## Your Task:")
        lines.append("Position each layer at its intended section. Validate no gaps > 2 bars.")
        lines.append("Call complete when the arrangement is solid.")
        return "\n".join(lines)
