# -*- coding: utf-8 -*-
import re
from .rule_optimizer import rule_optimize, _PHRASE_RES
from .hallucination_guard import prepare_prompt, validate_financial_response, ValidationResult
from . import response_cache
from .prompt_cache import cache_get, cache_set, cache_stats
from .token_counter import count_tokens

# Pre-compiled financial abbreviations (safe at any aggressiveness level —
# every substitution is a universally accepted acronym for the full term)
_FINANCE_ABBREV = [
    (re.compile(r'\bearnings per share\b', re.I),                        'EPS'),
    (re.compile(r'\bprice[- ]to[- ]earnings(?: ratio)?\b', re.I),       'P/E'),
    (re.compile(r'\bprice[- ]earnings(?: ratio)?\b', re.I),              'P/E'),
    (re.compile(r'\bprice[- ]to[- ]earnings[- ]growth\b', re.I),        'PEG'),
    (re.compile(r'\bdiscounted cash flow\b', re.I),                      'DCF'),
    (re.compile(r'\bfree cash flow\b', re.I),                            'FCF'),
    (re.compile(r'\bearnings before interest,?\s*taxes,?\s*depreciation,?\s*and\s*amortization\b', re.I), 'EBITDA'),
    (re.compile(r'\bearnings before interest and taxes\b', re.I),        'EBIT'),
    (re.compile(r'\byear[- ]over[- ]year\b', re.I),                      'YoY'),
    (re.compile(r'\bquarter[- ]over[- ]quarter\b', re.I),                'QoQ'),
    (re.compile(r'\bcompound annual growth rate\b', re.I),               'CAGR'),
    (re.compile(r'\breturn on equity\b', re.I),                          'ROE'),
    (re.compile(r'\breturn on assets\b', re.I),                          'ROA'),
    (re.compile(r'\breturn on invested capital\b', re.I),                'ROIC'),
    (re.compile(r'\bnet asset value\b', re.I),                           'NAV'),
    (re.compile(r'\bdebt[- ]to[- ]equity(?: ratio)?\b', re.I),          'D/E'),
    (re.compile(r'\bmargin of safety\b', re.I),                          'MOS'),
    (re.compile(r'\bnon[- ]banking financial compan(?:y|ies)\b', re.I), 'NBFC'),
    (re.compile(r'\bnet interest margin\b', re.I),                       'NIM'),
    (re.compile(r'\bnon[- ]performing asset\b', re.I),                   'NPA'),
    (re.compile(r'\bforeign institutional investor\b', re.I),            'FII'),
    (re.compile(r'\bforeign portfolio investor\b', re.I),                'FPI'),
    (re.compile(r'\bsecurities and exchange board of india\b', re.I),   'SEBI'),
    (re.compile(r'\breserve bank of india\b', re.I),                     'RBI'),
    (re.compile(r'\bfederal reserve\b', re.I),                           'Fed'),
    (re.compile(r'\bgross domestic product\b', re.I),                    'GDP'),
    (re.compile(r'\bconsumer price index\b', re.I),                      'CPI'),
    (re.compile(r'\bgoods and services tax\b', re.I),                    'GST'),
    (re.compile(r'\bmergers? and acquisitions?\b', re.I),                'M&A'),
    (re.compile(r'\bengineering,?\s*procurement,?\s*(?:and\s*)?construction\b', re.I), 'EPC'),
    (re.compile(r'\bprofit after tax\b', re.I),                          'PAT'),
    (re.compile(r'\bprofit before tax\b', re.I),                         'PBT'),
    (re.compile(r'\binitial public offering\b', re.I),                   'IPO'),
]


def finance_optimize(text: str) -> str:
    """
    Light noise removal + financial abbreviations.
    Safe for all financial prompts — never compresses numbers or specific context.
    """
    # Step 1: light noise removal (filler words only)
    result = rule_optimize(text, aggressiveness="light")
    # Step 2: financial abbreviations (always safe)
    for pattern, abbrev in _FINANCE_ABBREV:
        result = pattern.sub(abbrev, result)
    return result
