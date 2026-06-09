from __future__ import annotations

"""Tax-mode, GST repair, and AI-trigger logic.

This module keeps the extraction orchestration thin by grouping the bank-specific
repair rules and tax normalization heuristics in one place.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional


def _extractor():
	from . import extractor as _extractor_module
	return _extractor_module


def _maybe_correct_brokerage_from_total_and_tax(row: Dict[str, str]) -> Dict[str, str]:
	"""Correct brokerage only when total and tax strongly imply a valid 18% GST relationship."""
	extractor = _extractor()
	result = dict(row)
	total = extractor._amount_to_float(result.get("Total Inv Amt", ""))
	brok = extractor._amount_to_float(result.get("BROKERAGE Amount", ""))
	gst_total = extractor._amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = extractor._amount_to_float(result.get("CGST @ 9%", ""))
	sgst = extractor._amount_to_float(result.get("SGST @ 9%", ""))
	utgst = extractor._amount_to_float(result.get("UTGST", ""))
	igst = extractor._amount_to_float(result.get("IGST", ""))

	state_component = sgst if sgst > 0 else utgst
	tax_component = igst if igst > 0 else (cgst + state_component)
	if tax_component <= 0 and gst_total > 0:
		tax_component = gst_total
	elif tax_component > 0 and gst_total > 0 and abs(gst_total - tax_component) <= max(2.0, tax_component * 0.05):
		tax_component = gst_total

	if total <= 0 or tax_component <= 0:
		return result

	implied_brok = total - tax_component
	if implied_brok <= 0:
		return result

	tax_ratio = tax_component / implied_brok
	if tax_ratio < 0.16 or tax_ratio > 0.20:
		return result

	if brok > 0 and abs(brok - implied_brok) <= max(2.0, brok * 0.02):
		return result

	if brok > 0:
		current_total = brok + tax_component
		if abs(total - current_total) <= max(2.0, current_total * 0.02):
			return result

	result["BROKERAGE Amount"] = extractor._fmt_amount(implied_brok)
	return result


def _maybe_fix_ujjivan_gst_total(row: Dict[str, str]) -> Dict[str, str]:
	"""Fix Ujjivan rows where GST TOTAL is mis-mapped but tax components + total are coherent."""
	extractor = _extractor()
	result = dict(row)
	source_low = (result.get("Source File", "") or "").strip().lower()
	if "ujjivan" not in source_low:
		return result

	brok = extractor._amount_to_float(result.get("BROKERAGE Amount", ""))
	total = extractor._amount_to_float(result.get("Total Inv Amt", ""))
	gst_total = extractor._amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = extractor._amount_to_float(result.get("CGST @ 9%", ""))
	sgst = extractor._amount_to_float(result.get("SGST @ 9%", ""))
	utgst = extractor._amount_to_float(result.get("UTGST", ""))
	igst = extractor._amount_to_float(result.get("IGST", ""))

	state_component = sgst if sgst > 0 else utgst
	tax_component = igst if igst > 0 else (cgst + state_component)
	if brok <= 0 or total <= 0 or gst_total <= 0 or tax_component <= 0:
		return result

	expected_total = brok + tax_component
	if abs(total - expected_total) > max(2.0, expected_total * 0.02):
		return result

	if abs(gst_total - tax_component) <= max(2.0, tax_component * 0.02):
		return result

	result["GST TOTAL AMT"] = extractor._fmt_amount(tax_component)
	return result


def _maybe_fix_city_union_gst_total(row: Dict[str, str]) -> Dict[str, str]:
	"""Fix City Union rows where GST TOTAL is captured as roughly half of state-tax sum."""
	extractor = _extractor()
	result = dict(row)
	source_low = (result.get("Source File", "") or "").strip().lower()
	if "city union bank" not in source_low:
		return result

	brok = extractor._amount_to_float(result.get("BROKERAGE Amount", ""))
	total = extractor._amount_to_float(result.get("Total Inv Amt", ""))
	gst_total = extractor._amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = extractor._amount_to_float(result.get("CGST @ 9%", ""))
	sgst = extractor._amount_to_float(result.get("SGST @ 9%", ""))
	utgst = extractor._amount_to_float(result.get("UTGST", ""))
	igst = extractor._amount_to_float(result.get("IGST", ""))

	state_component = sgst if sgst > 0 else utgst
	state_sum = cgst + state_component
	if igst > 0 or brok <= 0 or total <= 0 or gst_total <= 0 or cgst <= 0 or state_component <= 0:
		return result
	if abs(cgst - state_component) > max(2.0, max(cgst, state_component) * 0.05):
		return result

	if abs(total - (brok + state_sum)) > max(2.0, total * 0.01):
		return result

	ratio = gst_total / state_sum if state_sum > 0 else 0.0
	if ratio < 0.35 or ratio > 0.65:
		return result

	result["GST TOTAL AMT"] = extractor._fmt_amount(state_sum)
	return result


def _maybe_fix_dhanlaxmi_tax_components(row: Dict[str, str]) -> Dict[str, str]:
	"""Repair Dhanlaxmi rows where one state-tax component is missing/corrupted but GST total is coherent."""
	extractor = _extractor()
	result = dict(row)
	source_low = (result.get("Source File", "") or "").strip().lower()
	if "dhanlaxmi bank" not in source_low:
		return result

	brok = extractor._amount_to_float(result.get("BROKERAGE Amount", ""))
	total = extractor._amount_to_float(result.get("Total Inv Amt", ""))
	gst_total = extractor._amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = extractor._amount_to_float(result.get("CGST @ 9%", ""))
	sgst = extractor._amount_to_float(result.get("SGST @ 9%", ""))
	utgst = extractor._amount_to_float(result.get("UTGST", ""))
	igst = extractor._amount_to_float(result.get("IGST", ""))

	if igst > 0 or gst_total <= 0:
		return result
	if brok > 0 and total > 0:
		tax_from_total = total - brok
		if tax_from_total <= 0 or abs(tax_from_total - gst_total) > max(2.0, gst_total * 0.02):
			return result

	tol = max(2.0, gst_total * 0.02)
	half = gst_total / 2.0
	state_component = sgst if sgst > 0 else utgst

	if cgst <= 0 and state_component > 0 and abs(gst_total - (2.0 * state_component)) <= tol:
		cgst = state_component
	elif state_component <= 0 and cgst > 0 and abs(gst_total - (2.0 * cgst)) <= tol:
		sgst = cgst
		utgst = 0.0
	elif cgst > 0 and state_component > 0 and abs(cgst - state_component) > max(2.0, max(cgst, state_component) * 0.10):
		close_to_half = (abs(cgst - half) <= tol) or (abs(state_component - half) <= tol)
		if close_to_half:
			cgst = half
			sgst = half
			utgst = 0.0

	if cgst > 0:
		result["CGST @ 9%"] = extractor._fmt_amount(cgst)
	if sgst > 0:
		result["SGST @ 9%"] = extractor._fmt_amount(sgst)
		result["UTGST"] = ""
	elif utgst > 0:
		result["UTGST"] = extractor._fmt_amount(utgst)
		result["SGST @ 9%"] = ""

	return result


def apply_confident_math_fill(row: Dict[str, str]) -> Dict[str, str]:
	"""Fill GST/total fields from arithmetic only when brokerage is validated against 18% tax math."""
	extractor = _extractor()
	result = dict(row)
	brok = extractor._amount_to_float(result.get("BROKERAGE Amount", ""))
	if brok <= 0:
		return result

	mode = extractor._detect_clear_tax_mode(result)
	if mode == "none":
		return result

	expected_gst = brok * 0.18
	expected_total = brok + expected_gst
	tol_gst = max(2.0, expected_gst * 0.02)
	tol_total = max(2.0, expected_total * 0.02)

	total = extractor._amount_to_float(result.get("Total Inv Amt", ""))
	gst_total = extractor._amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = extractor._amount_to_float(result.get("CGST @ 9%", ""))
	sgst = extractor._amount_to_float(result.get("SGST @ 9%", ""))
	utgst = extractor._amount_to_float(result.get("UTGST", ""))
	igst = extractor._amount_to_float(result.get("IGST", ""))

	brokerage_confident = False
	if total > 0 and abs(total - expected_total) <= tol_total:
		brokerage_confident = True
	if gst_total > 0 and abs(gst_total - expected_gst) <= tol_gst:
		brokerage_confident = True

	if mode == "igst" and igst > 0 and abs(igst - expected_gst) <= tol_gst:
		brokerage_confident = True
	if mode == "state":
		state_component = sgst if sgst > 0 else utgst
		half = expected_gst / 2.0
		tol_half = max(2.0, half * 0.02)
		if cgst > 0 and state_component > 0:
			if abs(cgst - half) <= tol_half and abs(state_component - half) <= tol_half:
				brokerage_confident = True

	if not brokerage_confident:
		return result

	if mode == "igst":
		result["CGST @ 9%"] = ""
		result["SGST @ 9%"] = ""
		result["UTGST"] = ""
		if igst <= 0:
			result["IGST"] = extractor._fmt_amount(expected_gst)
			igst = expected_gst
		if gst_total <= 0:
			result["GST TOTAL AMT"] = extractor._fmt_amount(igst if igst > 0 else expected_gst)
	elif mode == "state":
		half = expected_gst / 2.0
		result["IGST"] = ""
		if cgst <= 0:
			result["CGST @ 9%"] = extractor._fmt_amount(half)
		if sgst <= 0 and utgst <= 0:
			result["SGST @ 9%"] = extractor._fmt_amount(half)
			result["UTGST"] = ""
		elif sgst > 0 and utgst <= 0:
			result["SGST @ 9%"] = extractor._fmt_amount(sgst)
			result["UTGST"] = ""
		elif utgst > 0 and sgst <= 0:
			result["UTGST"] = extractor._fmt_amount(utgst)
			result["SGST @ 9%"] = ""

		cgst_f = extractor._amount_to_float(result.get("CGST @ 9%", ""))
		state_component_f = extractor._amount_to_float(result.get("SGST @ 9%", ""))
		if state_component_f <= 0:
			state_component_f = extractor._amount_to_float(result.get("UTGST", ""))
		state_sum = cgst_f + state_component_f
		if gst_total <= 0 and state_sum > 0:
			result["GST TOTAL AMT"] = extractor._fmt_amount(state_sum)

	gst_after_fill = extractor._amount_to_float(result.get("GST TOTAL AMT", ""))
	if total <= 0 and gst_after_fill > 0:
		result["Total Inv Amt"] = extractor._fmt_amount(brok + gst_after_fill)

	return result


def apply_gst_autofill(row: Dict[str, str]) -> Dict[str, str]:
	"""Auto-fill GST components from Brokerage when brokerage and GST total strongly imply 18% tax."""
	extractor = _extractor()
	result = dict(row)
	brok = extractor._amount_to_float(result.get("BROKERAGE Amount", ""))
	if brok <= 0:
		return result
	gst_total = extractor._amount_to_float(result.get("GST TOTAL AMT", ""))
	cgst = extractor._amount_to_float(result.get("CGST @ 9%", ""))
	sgst = extractor._amount_to_float(result.get("SGST @ 9%", ""))
	utgst = extractor._amount_to_float(result.get("UTGST", ""))
	igst = extractor._amount_to_float(result.get("IGST", ""))

	expected_gst = brok * 0.18
	tol = max(2.0, expected_gst * 0.05)

	if gst_total <= 0 or abs(gst_total - expected_gst) > tol:
		return result

	is_valid, _ = extractor.validate_math_extraction(result)
	if is_valid:
		return result

	has_state = (cgst > 0 or sgst > 0 or utgst > 0)

	if has_state:
		half = brok * 0.09
		result["CGST @ 9%"] = extractor._fmt_amount(half)
		if sgst > 0 or (not sgst and not utgst):
			result["SGST @ 9%"] = extractor._fmt_amount(half)
			result["UTGST"] = ""
		else:
			result["UTGST"] = extractor._fmt_amount(half)
		result["IGST"] = ""
		result["GST TOTAL AMT"] = extractor._fmt_amount(half * 2.0)
	else:
		result["IGST"] = extractor._fmt_amount(expected_gst)
		result["CGST @ 9%"] = ""
		result["SGST @ 9%"] = ""
		result["UTGST"] = ""
		result["GST TOTAL AMT"] = extractor._fmt_amount(expected_gst)

	return result


def enforce_tax_mode_fields(row: Dict[str, str]) -> Dict[str, str]:
	"""Normalize row to a single tax mode by zeroing the non-selected mode."""
	extractor = _extractor()
	clear_mode = extractor._detect_clear_tax_mode(row)
	if clear_mode == "none":
		brok = extractor._amount_to_float(row.get("BROKERAGE Amount", ""))
		balic_state = extractor.normalize_state_name((row.get("BALIC STATE", "") or "").strip())
		broker_state = extractor.normalize_state_name((row.get("BROKER GSTN STATE", "") or "").strip())
		if brok > 0 and balic_state and broker_state:
			clear_mode = "state" if balic_state.lower() == broker_state.lower() else "igst"
		else:
			clear_mode = extractor._infer_tax_mode_fallback(row)
	if clear_mode == "none":
		return row

	brok = extractor._amount_to_float(row.get("BROKERAGE Amount", ""))
	balic_state = extractor.normalize_state_name((row.get("BALIC STATE", "") or "").strip())
	broker_state = extractor.normalize_state_name((row.get("BROKER GSTN STATE", "") or "").strip())
	states_known = bool(balic_state and broker_state)
	states_match = states_known and balic_state.lower() == broker_state.lower()
	states_differ = states_known and balic_state.lower() != broker_state.lower()
	cgst = extractor._amount_to_float(row.get("CGST @ 9%", ""))
	sgst = extractor._amount_to_float(row.get("SGST @ 9%", ""))
	utgst = extractor._amount_to_float(row.get("UTGST", ""))
	igst = extractor._amount_to_float(row.get("IGST", ""))
	gst_total = extractor._amount_to_float(row.get("GST TOTAL AMT", ""))
	state_component = max(sgst, utgst)
	state_tax = cgst + state_component

	mode = clear_mode
	if mode == "igst":
		row["CGST @ 9%"] = ""
		row["SGST @ 9%"] = ""
		row["UTGST"] = ""
		if igst <= 0 and brok > 0 and (states_differ or states_known):
			igst = brok * 0.18
		if igst <= 0 and gst_total > 0:
			igst = gst_total
		if igst > 0:
			row["IGST"] = extractor._fmt_amount(igst)
		else:
			row["IGST"] = ""
		if gst_total <= 0 and igst > 0:
			row["GST TOTAL AMT"] = extractor._fmt_amount(igst)
	elif mode == "state":
		row["IGST"] = ""
		if cgst <= 0 and brok > 0 and states_match:
			cgst = brok * 0.09
		if sgst <= 0 and utgst <= 0 and brok > 0 and states_match:
			sgst = brok * 0.09
		if sgst <= 0 and utgst > 0:
			sgst = utgst
			utgst = 0.0
		if cgst <= 0 and sgst <= 0 and gst_total > 0:
			half = gst_total / 2.0
			cgst = half
			sgst = half
			utgst = 0.0
		if cgst > 0:
			row["CGST @ 9%"] = extractor._fmt_amount(cgst)
		else:
			row["CGST @ 9%"] = ""
		if sgst > 0:
			row["SGST @ 9%"] = extractor._fmt_amount(sgst)
		else:
			row["SGST @ 9%"] = ""
		row["UTGST"] = ""
		state_total = cgst + max(sgst, utgst)
		if gst_total <= 0 and state_total > 0:
			row["GST TOTAL AMT"] = extractor._fmt_amount(state_total)
	return row


def should_prioritize_ai_first(source_path: Optional[Path], source_display: str, page_text: str) -> bool:
	low = f"{str(source_path or '')} {source_display} {page_text[:1200]}".lower()
	triggers = [
		"city union",
		"city union bank",
		"probitas",
		"probitas insurance brokers",
		"turtlemint",
		"trusttech",
		"capri",
		"catalyst",
		"axis",
		"dbs",
		"motilal",
	]
	return any(token in low for token in triggers)


def _looks_mapping_incomplete(row: Dict[str, str]) -> bool:
	if not (row.get("Agent PAN", "") or "").strip():
		return True
	if not (row.get("Name of Service Receipient", "") or "").strip():
		return True
	if not (row.get("BROKER GSTN", "") or "").strip():
		return True
	if not (row.get("BALIC STATE", "") or "").strip():
		return True
	if not (row.get("Vendor Inv Date", "") or "").strip():
		return True
	if not (row.get("Vendor Inv No", "") or "").strip():
		return True
	if not (row.get("Total Inv Amt", "") or "").strip():
		return True
	if not (row.get("GST TOTAL AMT", "") or "").strip():
		return True
	if not (row.get("Narration", "") or "").strip():
		return True
	return False


def _has_gstin_role_conflict(row: Dict[str, str]) -> bool:
	"""Detect BALIC/BROKER GSTIN role collisions and obvious broker mis-maps."""
	balic_gstn = (row.get("BALIC GSTN", "") or "").strip().upper()
	broker_gstn = (row.get("BROKER GSTN", "") or "").strip().upper()
	if balic_gstn and broker_gstn and balic_gstn == broker_gstn:
		return True

	recipient_low = (row.get("Name of Service Receipient", "") or "").strip().lower()
	agent_pan = (row.get("Agent PAN", "") or "").strip().upper()
	if broker_gstn and ("bajaj" in recipient_low or "allianz" in recipient_low):
		if "AADCA1701E" in broker_gstn:
			return True
		if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", agent_pan) and agent_pan not in broker_gstn:
			return True

	return False


def _looks_mapping_improbable(row: Dict[str, str]) -> bool:
	"""Detect suspicious numeric mapping where LLM rescue is useful."""
	extractor = _extractor()
	normalized = enforce_tax_mode_fields(dict(row))

	if _has_gstin_role_conflict(normalized):
		return True

	is_math_valid, _ = extractor.validate_math_extraction(normalized)
	if not is_math_valid:
		return True

	try:
		brok = float((normalized.get("BROKERAGE Amount", "") or "").replace(",", "")) if (normalized.get("BROKERAGE Amount", "") or "").strip() else 0.0
		total = float((normalized.get("Total Inv Amt", "") or "").replace(",", "")) if (normalized.get("Total Inv Amt", "") or "").strip() else 0.0
		cgst = float((normalized.get("CGST @ 9%", "") or "").replace(",", "")) if (normalized.get("CGST @ 9%", "") or "").strip() else 0.0
		sgst = float((normalized.get("SGST @ 9%", "") or "").replace(",", "")) if (normalized.get("SGST @ 9%", "") or "").strip() else 0.0
		utgst = float((normalized.get("UTGST", "") or "").replace(",", "")) if (normalized.get("UTGST", "") or "").strip() else 0.0
		igst = float((normalized.get("IGST", "") or "").replace(",", "")) if (normalized.get("IGST", "") or "").strip() else 0.0
		gst_total = float((normalized.get("GST TOTAL AMT", "") or "").replace(",", "")) if (normalized.get("GST TOTAL AMT", "") or "").strip() else 0.0

		state_tax = max(sgst, utgst)

		if igst > 0 and (cgst > 0 or state_tax > 0):
			return True
		if brok > 0 and total > 0 and total < brok:
			return True
		if gst_total > 0:
			expected_gst = igst if igst > 0 else (cgst + state_tax)
			if expected_gst > 0 and abs(gst_total - expected_gst) > max(2.0, expected_gst * 0.05):
				return True
		if brok > 0:
			tax_component = igst if igst > 0 else (cgst + state_tax)
			if tax_component > 0:
				ratio = tax_component / brok
				if ratio < 0.05 or ratio > 0.30:
					return True
			if gst_total > 0:
				gst_ratio = gst_total / brok
				if gst_ratio < 0.05 or gst_ratio > 0.30:
					return True
		if gst_total >= 900000:
			return True
	except (ValueError, TypeError, AttributeError, ZeroDivisionError):
		return True

	return False


def _looks_gst_mode_ambiguous(row: Dict[str, str]) -> bool:
	"""Detect rows where tax values exist but GST mode is not clearly resolvable."""
	extractor = _extractor()
	total = extractor._amount_to_float(row.get("Total Inv Amt", ""))
	brok = extractor._amount_to_float(row.get("BROKERAGE Amount", ""))
	gst_total = extractor._amount_to_float(row.get("GST TOTAL AMT", ""))
	cgst = extractor._amount_to_float(row.get("CGST @ 9%", ""))
	sgst = extractor._amount_to_float(row.get("SGST @ 9%", ""))
	utgst = extractor._amount_to_float(row.get("UTGST", ""))
	igst = extractor._amount_to_float(row.get("IGST", ""))
	mode = extractor._detect_clear_tax_mode(row)
	has_tax_bits = any(v > 0 for v in [cgst, sgst, utgst, igst])
	if gst_total <= 0:
		return False
	if mode == "none":
		return True
	if mode in {"igst", "state"} and has_tax_bits:
		if mode == "igst" and (cgst > 0 or sgst > 0 or utgst > 0):
			return True
		if mode == "state" and igst > 0:
			return True
		if brok > 0 and total > 0:
			expected_total = brok + gst_total
			if abs(total - expected_total) > max(2.0, expected_total * 0.02):
				return True
		return False
	if brok > 0 and total > 0:
		return True
	return False


def _gst_mode_context_hint(rows: List[Dict[str, str]]) -> str:
	"""Summarize ambiguous GST rows for LLM fallback."""
	extractor = _extractor()
	parts: List[str] = []
	for idx, row in enumerate(rows, start=1):
		brok = extractor._amount_to_float(row.get("BROKERAGE Amount", ""))
		total = extractor._amount_to_float(row.get("Total Inv Amt", ""))
		gst_total = extractor._amount_to_float(row.get("GST TOTAL AMT", ""))
		cgst = extractor._amount_to_float(row.get("CGST @ 9%", ""))
		sgst = extractor._amount_to_float(row.get("SGST @ 9%", ""))
		utgst = extractor._amount_to_float(row.get("UTGST", ""))
		igst = extractor._amount_to_float(row.get("IGST", ""))
		mode = extractor._detect_clear_tax_mode(row)
		parts.append(
			f"row{idx}: Brokerage={extractor._fmt_amount(brok) if brok > 0 else 'blank'}, "
			f"Total={extractor._fmt_amount(total) if total > 0 else 'blank'}, "
			f"GST TOTAL={extractor._fmt_amount(gst_total) if gst_total > 0 else 'blank'}, "
			f"CGST={extractor._fmt_amount(cgst) if cgst > 0 else '0'}, SGST={extractor._fmt_amount(sgst) if sgst > 0 else '0'}, "
			f"UTGST={extractor._fmt_amount(utgst) if utgst > 0 else '0'}, IGST={extractor._fmt_amount(igst) if igst > 0 else '0'}, mode={mode}"
		)
	return " | ".join(parts)


__all__ = [
	"_maybe_correct_brokerage_from_total_and_tax",
	"_maybe_fix_ujjivan_gst_total",
	"_maybe_fix_city_union_gst_total",
	"_maybe_fix_dhanlaxmi_tax_components",
	"apply_confident_math_fill",
	"apply_gst_autofill",
	"enforce_tax_mode_fields",
	"should_prioritize_ai_first",
	"_looks_mapping_incomplete",
	"_has_gstin_role_conflict",
	"_looks_mapping_improbable",
	"_looks_gst_mode_ambiguous",
	"_gst_mode_context_hint",
]
