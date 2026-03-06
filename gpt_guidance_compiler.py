#!/usr/bin/env python3
# -*- coding: utf-8 -*-
   

from __future__ import annotations

import json
from typing import Any, Dict, List

from jsonschema import validate


                              
                                 
                              
                                                                                           
                                                                         
OPENAI_MODEL = "gpt-5.2"
OPENAI_REASONING_EFFORT = "medium"
OPENAI_TEXT_VERBOSITY = "low"
OPENAI_MAX_OUTPUT_TOKENS = 1024


def extract_targets_data_from_meta_dict(meta: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
                                                                         
    return {k: meta[k] for k in keys if k in meta}


def _response_to_text(resp: Any) -> str:
                                                                             
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

                                              
    try:
        out: List[str] = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for c in getattr(item, "content", []) or []:
                ctype = getattr(c, "type", None)
                if ctype in ("output_text", "text"):
                    out.append(getattr(c, "text", "") or "")
        return "".join(out)
    except Exception:
        return ""


def call_and_validate(llm: Any, messages: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
       
    fallback = {"clauses": [], "aggregator": "none"}

    try:
        resp = llm.responses.create(
            model=OPENAI_MODEL,
            input=messages,
            reasoning={"effort": OPENAI_REASONING_EFFORT},
            text={
                "verbosity": OPENAI_TEXT_VERBOSITY,
                "format": {"type": "json_object"},
            },
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )

        resp_text = _response_to_text(resp)
        if not resp_text.strip():
            return fallback

        data = json.loads(resp_text)

                                             
        if "aggregator" in data and data["aggregator"] is None:
            data["aggregator"] = "none"

        validate(instance=data, schema=schema)
        return data

    except Exception:
        return fallback


def prepare_prompt_for_guidance_parsing(
    guidance_text: str,
    all_targets: List[str],
    schema: Dict[str, Any],
) -> List[Dict[str, Any]]:
    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant that parses a heuristic into one or more residual clauses and chooses how to aggregate them.\n"
            "Each clause corresponds to one atomic residual and its target variables.\n"
            "If multiple clauses, determine the aggregator based on the logical connectives in the heuristic (e.g., 'and', 'or', 'if...then').\n"
            "Logical connectives may be implicitly indicated: e.g., if the heuristic says change, it means increase 'or' decrease'. \n"
            "\n"
            "Atomic residual templates (short name => meaning):\n"
            " inc(x,k): recommend increasing x by k (delta step)\n"
            " dec(x, k): recommend decreasing x by k (delta step).\n"
            " eq_const(x,v): enforce x == v when setting a variable x to a literal constant v (number/string) or a categorical label.\n"
            " eq_var(x,y): enforce x == y when setting two variables to be equal (both sides must be variables).\n"
            " in_box(x,L,U): enforce L <= x <= U\n"
            " diff_ge(x,y,δ): enforce y + δ >= x\n"
            " ratio_eq(x,y,r): enforce x/y ≈ r\n"
            " sum_le([x_i],k): enforce Σx_i <= k\n"
            " monotone([x_i],sense): enforce sequence monotonicity\n\n"
            "Target selection rules:\n"
            " - Arity must match the template notation:\n"
            "   x-only templates (inc, dec, eq_const, in_box) => targets has exactly 1 variable (the x).\n"
            "   x,y templates (eq_var, diff_ge, ratio_eq) => targets has exactly 2 variables (the x and y).\n"
            "   [x_i] templates (sum_le, monotone) => targets has 1 or more variables.\n\n"
            "Return exactly one JSON object matching this schema:\n"
            + json.dumps(schema, indent=2)
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f'Heuristic: "{guidance_text}"\n'
            f"Candidate variables: {all_targets}\n\n"
            "Break the heuristic into clauses. For each clause, specify which residual template and the list of variables it applies to. "
            "Then choose an aggregator to combine these clauses. Return only the JSON."
            "If the heuristic is empty or you cannot understand or map any part of it to a residual, return an empty JSON."
        ),
    }
    return [system_msg, user_msg]


def prepare_prompt_to_get_residuals_from_parsed_guidance(
    step1_output: Dict[str, Any],
    guidance_text: str,
    schema: Dict[str, Any],
    targets_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    atomic_residuals_meaning: Dict[str, str] = {
        "inc": (
            "inc(k): increase the target variable by a SIGNED delta k.\n"
            'JSON parameters MUST be exactly: {"k": <float>} with k > 0.\n'
        ),
        "dec": (
            "dec(k): decrease the target variable by a SIGNED delta k.\n"
            'JSON parameters MUST be exactly: {"k": <float>} with k < 0.\n'
        ),
        "eq_const": (
            "eq_const(v): set the target variable equal to a constant (number/string/category).\n"
            'JSON parameters MUST be exactly: {"v": <value>}.\n'
        ),
        "eq_var": (
            "eq_var(y): set the target variable equal to ANOTHER VARIABLE y (not a constant).\n"
            'JSON parameters MUST be exactly: {"y": "<other_var_name>"}.\n'
        ),
        "in_box": "in_box(x, L, U): enforce L <= x <= U. `L` and `U` are bounds.\n",
        "diff_ge": "diff_ge(x, y, delta): enforce y + delta >= x. `delta` is margin.\n",
        "ratio_eq": "ratio_eq(x, y, r): enforce x / y ≈ `r`.\n",
        "sum_le": "sum_le(list_of_x, k): enforce sum(x_i) <= `k`.\n",
        "monotone": "monotone(list_of_x, sense): sequence monotonicity; `sense` is 'inc' or 'dec'.\n",
    }

    residual_type = step1_output["residual_type"]
    system = {
        "role": "system",
        "content": (
            "You are given a heuristic and a clause parsed from it.\n"
            "The parsed clause consists of an atomic residual and the corresponding target variable(s).\n"
            "Use provided heuristic text to assign parameter values for the atomic residual.\n"
            "Use the provided meta data for the target variables to keep all numeric parameters within allowed ranges, and categorical parameters within allowed sets.\n"
            "Use the following atomic residual templates and their parameter names as guidance:\n\n"
            f"{residual_type}: {atomic_residuals_meaning[residual_type]}\n\n"
            "For each clause, populate `parameters` with the appropriate keys.\n"
            "Also include:\n"
            "  - `confidence`: your confidence (0–1) in the chosen parameters.\n"
            "  - `importance`: how strongly the heuristic drove this choice (0–1).\n"
            "  - `evidence`: ≤20 words copied from the heuristic supporting this clause.\n"
            "  - `assumption`: ≤20 words describing any assumptions you made.\n\n"
            "Return exactly one JSON object matching this schema.\n"
            + json.dumps(schema, indent=2)
        ),
    }
    user = {
        "role": "user",
        "content": (
            f'Heuristic: "{guidance_text}"\n'
            f"Clause: {json.dumps(step1_output, indent=2)}\n"
            f"Meta data for target variables: {json.dumps(targets_meta, indent=2)}\n\n"
            "For the residual type, assign parameter values using the heuristic.\n"
            "Keep parameter values within allowed range/set mentioned in meta data.\n"
            "Return only the JSON. No explanations."
            "If the heuristic is empty or the parsed information is empty, return an empty JSON."
        ),
    }
    return [system, user]


def run_guidance_micro_reasoners(
    llm: Any,
    predicate_info: List[Dict[str, Any]],
    predicate_guidance: Dict[str, str],
    targets_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    parsing_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "clauses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "residual_type": {
                            "type": "string",
                            "enum": [
                                "inc",
                                "dec",
                                "eq_const",
                                "eq_var",
                                "in_box",
                                "diff_ge",
                                "ratio_eq",
                                "sum_le",
                                "monotone",
                            ],
                        },
                        "targets": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["residual_type", "targets"],
                    "additionalProperties": False,
                },
                "minItems": 0,
            },
            "aggregator": {"type": ["string"], "enum": ["and", "or", "xor", "implication", "none"]},
        },
        "required": ["clauses", "aggregator"],
        "additionalProperties": False,
    }

    residual_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "clause": {
                "type": "object",
                "properties": {
                    "residual_type": {"type": "string"},
                    "targets": {"type": "array", "items": {"type": "string"}},
                    "parameters": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence": {"type": "string"},
                    "assumption": {"type": "string"},
                },
                "required": [
                    "residual_type",
                    "targets",
                    "parameters",
                    "confidence",
                    "importance",
                    "evidence",
                    "assumption",
                ],
                "additionalProperties": False,
            }
        },
        "required": ["clause"],
        "additionalProperties": False,
    }

    guidance_hints: List[Dict[str, Any]] = []

    for info in predicate_info:
        if not info.get("fired", False):
            continue

        pid = info["predicate"]
        guidance_text = predicate_guidance.get(pid, " ")

        parsing_prompt = prepare_prompt_for_guidance_parsing(
            guidance_text, list(targets_meta.keys()), parsing_schema
        )
        parsed_guidance = call_and_validate(llm, parsing_prompt, parsing_schema)
        parsed_clauses = parsed_guidance.get("clauses", [])

        if len(parsed_clauses) <= 1:
            parsed_guidance["aggregator"] = "none"

        residuals_info_list: List[Dict[str, Any]] = []
        for clause in parsed_clauses:
            specific_targets = extract_targets_data_from_meta_dict(targets_meta, clause["targets"])
            residual_prompt = prepare_prompt_to_get_residuals_from_parsed_guidance(
                clause, guidance_text, residual_schema, specific_targets
            )
            residuals_info = call_and_validate(llm, residual_prompt, residual_schema)
            residuals_info_list.append(residuals_info)

        merged: Dict[str, Any] = {
            "predicate": pid,
            "clauses": [],
            "aggregator": parsed_guidance["aggregator"],
        }

                                                                                        
        for base_clause in parsed_guidance.get("clauses", []):
            for res_info in residuals_info_list:
                detailed = (res_info or {}).get("clause", {})
                if not detailed:
                    continue
                if (
                    base_clause.get("residual_type") == detailed.get("residual_type")
                    and base_clause.get("targets") == detailed.get("targets")
                ):
                    merged["clauses"].append(detailed)
                    break

        guidance_hints.append(merged)

    return guidance_hints
