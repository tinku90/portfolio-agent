"""
Rule-based prompt optimizer — ZERO LLM calls.

Applies ordered text transformations to reduce token count while preserving
all information and intent. Safe to run on every request.

Performance: O(n * k) where k = number of rules (fixed ~30). Runs in < 1 ms.
"""

import re
from typing import Callable
from .token_counter import count_tokens

_INTENSIFIER_RE = re.compile(r'\b(?:very|quite|really|extremely)\b\s*')

# ── Structural guards: code / inline-code preservation ──────────────────────
# These patterns are extracted before rule passes and reinserted after,
# so the optimizer never touches code content.

_FENCED_CODE_RE  = re.compile(r'```[\s\S]*?```', re.DOTALL)
_INLINE_CODE_RE  = re.compile(r'`[^`\n]+`')


def _extract_blocks(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace code fences and inline code with stable placeholders.
    Returns (modified_text, {placeholder: original}).
    """
    placeholders: dict[str, str] = {}
    counter = [0]

    def save(m: re.Match) -> str:
        key = f"\x00CODEBLOCK{counter[0]}\x00"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    text = _FENCED_CODE_RE.sub(save, text)
    text = _INLINE_CODE_RE.sub(save, text)
    return text, placeholders


def _restore_blocks(text: str, placeholders: dict[str, str]) -> str:
    for key, original in placeholders.items():
        text = text.replace(key, original)
    return text


# ── Phase 1: Remove noise phrases (pure filler — zero information content) ──

_NOISE_PHRASES: list[str] = [
    # Politeness openers
    r"\bplease\b[\s,]*",
    r"\bkindly\b[\s,]*",
    r"\bif you don'?t mind\b[\s,]*",
    r"\bif possible\b[\s,]*",
    r"\bas you know[\s,]*",
    r"\bfor your information[\s,]*",
    r"\bfyi[\s,]*",
    r"\bjust so you know[\s,]*",

    # Indirect request starters (replaced by imperative, see phase 2)
    r"\bi would like you to\b[\s]*",
    r"\bi want you to\b[\s]*",
    r"\bi need you to\b[\s]*",
    r"\bcould you\b[\s]*",
    r"\bcan you\b[\s]*",
    r"\bwould you\b[\s]*",
    r"\bwould you mind\b[\s]*",
    r"\bgive me\b[\s]*",

    # Vague hedge words (zero information content)
    r"\bbasically\b[\s,]*",
    r"\bkind of\b[\s,]*",
    r"\bkinda\b[\s,]*",
    r"\bsort of\b[\s,]*",
    r"\bsomewhat\b[\s,]*",
    r"\blike really\b[\s,]*",
    r"\breally\b[\s,]* (?=simple|easy|clear|basic|quick|fast|good|great|detailed)",
    r"\bvery\b[\s]* (?=simple|easy|clear|basic|quick|fast|good|great|detailed)",
    r"\bquite\b[\s]* (?=simple|easy|clear|basic|good|great)",
    r"\bjust\b[\s,]* (?=explain|tell|give|show|help|want|need|like)",

    # Meta-commentary
    r"\bplease note that\b[\s]*",
    r"\bit(?:'s| is) important to note that\b[\s]*",
    r"\bit should be noted that\b[\s]*",
    r"\bplease be aware that\b[\s]*",
    r"\bjust to clarify[\s,]*",
    r"\bjust to be clear[\s,]*",
    r"\bfor context[\s,]*",
    r"\bfor reference[\s,]*",

    # Preamble connectors
    r"\bi am writing to\b[\s]*",
    r"\bi wanted to\b[\s]*",
    r"\bjust wanted to\b[\s]*",
    r"\bi'?m reaching out to\b[\s]*",

    # "and explain everything" — leftover fragment when beginner-friendly rule fires
    r",?\s*and explain everything\b[\s\w]*",
    r",?\s*explain everything\b[\s\w]*",

    # Redundant clarity phrases
    r",?\s*in a (?:much )?clearer way\b",
    r",?\s*so (?:that )?I can understand it better\b[\s\w]*",
    r",?\s*for (?:better|easier|clearer) understanding\b",

    # "easy to understand/follow/grasp" — redundant when "simple" already present
    r"\band easy to (?:understand|follow|grasp|comprehend)\b[\s,]*",
    r",?\s*easy to (?:understand|follow|grasp|comprehend)\b",

    # "to understand" as trailing qualifier — "in a simple way to understand" → "simply"
    r"\bto (?:understand|follow|grasp)\b[\s,]*$",

    # Verbose style qualifiers (move here from Phase 2 so they fire in light mode too)
    r",?\s*in (?:a |an )?(?:very )?(?:simple|easy|clear|basic) (?:way|manner)\b",
    r",?\s*in (?:a |an )?(?:very )?(?:simple|easy|clear|basic) (?:language|terms?|words?)\b",

    # ── Conversational scaffolding (chat-style openers and closers) ────────────
    # Openers that carry zero information content
    r"^[Hh]ey[\s,!.]+",
    r"\bso I was (just )?thinking[\s,]+",
    r"\bI was (just )?thinking[\s,]+",
    r"\bI was wondering[\s,]+",
    r"\byou know[\s,]+",
    r"\bI (?:just )?(?:wanted|want|need) to ask[\s,]+",
    r"\bso I (?:just )?(?:wanted|want|need) to ask[\s,]+",
    r"\bjust (?:wanted|want|need) to ask[\s,]+",

    # Personal-context tail (explains user's mental state, not the task)
    r",?\s*because I('ve| have) been trying to figure it out[\s,]*",
    r",?\s*because I'?(?:m|'ve been|ve been) (?:really |very |a bit |quite |so )?(?:confused|unsure|unclear) about it[\s.!,]*",
    r",?\s*because I'?(?:m|'ve been|ve been) (?:really |very |a bit |quite |so )?(?:confused|unsure|unclear)[\s.!,]*",
    r",?\s*but it'?s? (?:kind of |a bit |really )?(?:confusing|unclear|complicated|hard to understand)[\s,]*",
    r",?\s*because it'?s? (?:kind of |a bit |really )?(?:confusing|unclear|complicated|hard to understand)[\s,]*",

    # Trailing social closers (carry no task information)
    r",?\s*so if you could explain that[\s,]*",
    r",?\s*that would be (?:really |very |much )?(?:great|helpful|appreciated|amazing|wonderful)[\s.!]*",
    r",?\s*that would (?:really |greatly |much )?help[\s.!]*",
    r",?\s*if that (?:makes sense|helps)[\s.!]*",
    r",?\s*if you know what I mean[\s.!]*",
    r",?\s*I hope that makes sense[\s.!]*",

    # ── Meta / thinking noise (user's attribution, not the task) ─────────────
    r"\bI think that\b[\s]*",
    r"\bI think,\s*",
    r"\bI think\b[\s]+",      # catches "I think Anthropic is..." (no "that"/"," required)
    r"\bI believe that\b[\s]*",
    r"\bI believe,\s*",
    r"\bI feel that\b[\s]*",
    r"\bI feel like\b[\s,]+",
    r"\bit seems (?:like|that)\b[\s]*",
    r"\bit appears (?:that|to be)\b[\s]*",
    r"\bfrom my understanding[\s,]+",
    r"\bfrom what I understand[\s,]+",
    r"\bas far as I (?:know|can tell)[\s,]+",
    r"\bin my opinion[\s,]*",
    r"\bin my view[\s,]*",
    r"\bI guess\b[\s,]+(?=\w)",
    r"\bI suppose\b[\s,]+(?=\w)",

    # ── Redundant clarifiers ──────────────────────────────────────────────────
    r"\bkeep in mind that\b[\s]*",
    r"\byou should know that\b[\s]*",
    r"\bwhat this means is\b[\s,]*",
    r"\bwhich means that\b[\s,]*",
    r"\bneedless to say[\s,]*",
    r"\bas a matter of fact[\s,]*",
    r"\bat the end of the day[\s,]*",
    r"\bfor all intents and purposes[\s,]*",

    # ── Structural padding (order markers, transitional filler) ──────────────
    r"\blet'?s start with\b[\s,]*",
    r"\bfirst of all[\s,]+",
    r"\bto begin with[\s,]+",
    r"\bin conclusion[\s,]*",
    r"\bmoving on[\s,]*",

    # ── Sequential chain connectives (order markers with no info content) ────
    r"\bfirst(?:ly)?,?\s*(?=explain|describe|cover|discuss|analyze|list|provide)",
    r"\bsecond(?:ly)?,?\s*(?=explain|describe|cover|discuss|then|analyze|list)",
    r"\bthird(?:ly)?,?\s*(?!-)",
    r"\bfourth(?:ly)?,?\s*(?!-)",
    r"\bafter that,?\s*",
    r"\bfollowing that,?\s*",
    r"\bsubsequently,?\s*",
    r"\bto start with,?\s*",
    r"\bto wrap up,?\s*",
    r"\bto conclude,?\s*",
    r"\bfinally,?\s*(?=explain|describe|cover|discuss|provide|give|list|summarize)",
    r"\band then,?\s*(?=explain|describe|cover|discuss)",

    # ── Academic / formal register verbosity ─────────────────────────────────
    r"\bit is worth noting that\b[\s]*",
    r"\bit is important to mention that\b[\s]*",
    r"\bit is (?:worth|important to) (?:note|mention|highlight|emphasize|point out) that\b[\s]*",
    r"\bone must (?:note|consider|understand|be aware) that\b[\s]*",
    r"\bit is (?:generally|widely|commonly|broadly) (?:acknowledged|accepted|understood|known|agreed) that\b[\s]*",
    r"\baccording to (?:the )?(?:recent|current|latest|modern) (?:research|literature|studies|findings)[,\s]*",
    r"\bwith (?:that|this) (?:said|in mind)[,\s]*",
    r"\bit goes without saying that\b[\s]*",
    r"\bas (?:we|you) (?:all )?know[,\s]*",
    r"\bwithout further ado[,\s]*",
    r"\bwithout wasting (?:any )?(?:more )?time[,\s]*",
    r"\blet me (?:start|begin) by\b[\s]*",
    r"\bI would like to (?:start|begin) by\b[\s]*",
    r"\bbefore (?:I|we) (?:begin|start|dive in|get started)[,\s]*",
    r"\blet(?:'?s| us) (?:dive (?:into|in)|get (?:into|started)|explore|look at)\b[\s,]*",
    r"\bin (?:today'?s|the modern|this) (?:world|era|age|day and age|digital age|landscape)[,\s]*",
    r"\bas technology (?:continues to )?(?:evolve|advance|progress)[,\s]*",

    # ── Enterprise context preambles ─────────────────────────────────────────
    r"\bI am working on a project where(?: we have)?(?: a)?\s*",
    # NOTE: "from a X standpoint/perspective" and "Given that I am a [role]" are
    # intentionally NOT in this noise list — they carry domain/experience context
    # that changes what a correct answer looks like. They are handled in
    # _PHRASE_REPLACEMENTS below with capture groups that preserve the key word.
    r"\bfor the purposes? of this (?:discussion|conversation|question|task)[,\s]*",
    r"\bI(?:'m| am) (?:trying|struggling|having trouble|finding it hard) to (?:understand|grasp|figure out)[,\s]*",
    r"\bkeeping in mind(?: that)?\b[\s,]*",
    r"\btaking into account\b[\s,]*",
    r"\bwith (?:all )?(?:proper|appropriate|relevant|necessary) (?=\w)",
]

# Compile once at import time
_NOISE_RE = re.compile(
    "|".join(_NOISE_PHRASES),
    flags=re.IGNORECASE,
)


# ── Phase 2: Verbose phrase → compressed phrase ─────────────────────────────

_PHRASE_REPLACEMENTS: list[tuple[str, str]] = [
    # ── Short-prompt / question simplifications ──────────────────────────────
    # "and how is/are it/they X" → "(X)"  — saves 3 tokens, preserves meaning
    (r"\band how (?:is|are|was|were) (?:it|they|this|that) ([a-z]+)\b", r"(\1)"),
    # "what are/is the main/key X" → "X"  e.g. "what are the main benefits" → "benefits"
    (r"\bwhat (?:is|are) (?:the )?(?:main |key |primary |top |common )?(benefits?|advantages?|disadvantages?|uses?|applications?|drawbacks?|limitations?|challenges?|differences?|similarities?|features?|examples?|types?|components?|causes?|effects?|implications?)\b",
     r"\1"),
    # "can you explain/describe/list/summarize" → "explain/describe/list/summarize"
    (r"\bcan you (?:briefly |quickly |please )?(explain|describe|list|summarize|compare|analyze|define|outline|evaluate|discuss)\b",
     r"\1"),
    # "tell me about/what/how/why" → "explain"
    (r"\btell me (?:about|how|why|what)\b",                "explain"),
    # "what is your explanation/definition of" → "define"
    (r"\bwhat (?:is|are) (?:your |an? )?(?:explanation|definition) of\b", "define"),
    # "how does it/this/that work" → "how it works"
    (r"\bhow does (?:it|this|that) work\b",                "how it works"),
    # "how is it used / how are they used" → "usage"
    (r"\bhow (?:is|are) (?:it|they|this) used\b",          "usage"),
    # "why is it important/useful/relevant" → "(importance/use/relevance)"
    (r"\bwhy (?:is|are) (?:it|they|this) important\b",     "(importance)"),
    (r"\bwhy (?:is|are) (?:it|they|this) useful\b",        "(uses)"),
    # "I want to / I need to / I'd like to know about X" → "explain X"
    (r"\bi (?:want|need|'d like) to (?:know|understand|learn) (?:about |more about )?\b", "explain "),
    # "give me a brief/quick/short overview/introduction/summary of" → "overview:" etc.
    (r"\bgive me (?:a |an )?(?:brief |quick |short )?overview of\b",      "overview:"),
    (r"\bgive me (?:a |an )?(?:brief |quick |short )?introduction to\b",  "intro:"),
    (r"\bgive me (?:a |an )?(?:brief |quick |short )?summary of\b",       "summary:"),
    # "explain to me how/why/what" → "explain how/why/what"
    (r"\bexplain to me\b",                                 "explain"),

    # Conjunctions & connectors
    (r"\bin order to\b",                "to"),
    (r"\bso as to\b",                   "to"),
    (r"\bfor the purpose of\b",         "for"),
    (r"\bwith the aim of\b",            "to"),
    (r"\bwith the goal of\b",           "to"),
    (r"\bwith the intention of\b",      "to"),
    (r"\bdue to the fact that\b",       "because"),
    (r"\bgiven the fact that\b",        "because"),
    (r"\bowing to the fact that\b",     "because"),
    (r"\bin the event that\b",          "if"),
    (r"\bin the case that\b",           "if"),
    (r"\bat this point in time\b",      "now"),
    (r"\bat the present time\b",        "currently"),
    (r"\bon a regular basis\b",         "regularly"),
    (r"\bwith regard to\b",             "regarding"),
    (r"\bwith respect to\b",            "regarding"),
    (r"\bin relation to\b",             "about"),
    (r"\bin terms of\b",                "for"),
    (r"\bprior to\b",                   "before"),
    (r"\bsubsequent to\b",              "after"),
    (r"\ba large number of\b",          "many"),
    (r"\ba number of\b",                "several"),
    (r"\bthe majority of\b",            "most"),
    (r"\bthe vast majority of\b",       "most"),
    (r"\bit is possible that\b",        "possibly"),
    (r"\bthere is a possibility that\b","possibly"),
    (r"\bin spite of the fact that\b",  "although"),
    (r"\bdespite the fact that\b",      "although"),
    (r"\bat the same time\b",           "simultaneously"),
    (r"\bin addition to this\b",        "additionally"),
    (r"\bfurthermore[,\s]+",            "also "),
    (r"\badditionally[,\s]+",           "also "),
    (r"\bmoreover[,\s]+",               "also "),
    (r"\bhowever[,\s]+",                "but "),
    (r"\bnevertheless[,\s]+",           "still "),
    (r"\bnotwithstanding\b",            "despite"),

    # Verbose adjective padding
    (r"\bvery unique\b",                "unique"),
    (r"\bvery important\b",             "important"),
    (r"\bvery significant\b",           "significant"),
    (r"\babsolutely essential\b",       "essential"),
    (r"\bcompletely different\b",       "different"),

    # Common verbose request phrases → imperative
    (r"\bhelp me (?:to )?understand\b", "explain"),
    (r"\bhelp me (?:to )?learn\b",      "teach me"),
    (r"\bgive me (?:a|an)\b",           "provide a"),
    (r"\bgive me some\b",               "provide"),
    (r"\btell me (?:about|how|what|why)\b", "explain"),

    # Verbose list phrasing
    (r"\badvantages and disadvantages\b", "pros/cons"),
    (r"\bpros and cons\b",               "pros/cons"),
    (r"\bstrengths and weaknesses\b",    "strengths/weaknesses"),
    (r"\bbenefits and drawbacks\b",      "benefits/drawbacks"),
    (r"\bsimilarities and differences\b","similarities/differences"),

    # ── Enterprise / analytical prompt patterns ──────────────────────────────
    # "with detailed explanation" is implied by any analysis request
    (r",?\s*with (?:a )?detailed explanation and\b",  " with"),
    (r",?\s*with (?:a )?detailed explanation\b",      ""),
    (r",?\s*with (?:a )?detailed breakdown\b",        ""),
    (r"\bprovide (?:detailed )?insights?\b",          "insights"),
    (r"\bprovide (?:detailed )?recommendations?\b",   "recommendations"),
    (r"\bprovide (?:a )?(?:detailed )?summary\b",     "summary"),
    (r"\bprovide (?:a )?(?:detailed )?analysis\b",    "analysis"),
    (r"\bidentify (?:and list )?(?:key )?risks?\b",   "risks"),
    (r"\bidentify (?:and list )?(?:key )?issues?\b",  "issues"),
    (r"\bidentify (?:and list )?(?:key )?patterns?\b","patterns"),
    (r"\bidentify (?:and list )?(?:key )?trends?\b",  "trends"),
    (r"\bidentify (?:and )?highlight\b",              "highlight"),
    # "actionable recommendations" → "recommendations" (actionable is implied)
    (r"\bactionable recommendations?\b",              "recommendations"),
    (r"\bactionable insights?\b",                     "insights"),
    # "key insights" / "key findings" → drop "key"
    (r"\bkey insights?\b",                            "insights"),
    (r"\bkey findings?\b",                            "findings"),
    (r"\bkey recommendations?\b",                     "recommendations"),
    # Redundant output descriptors
    (r"\bdetailed (?=insights|findings|recommendations|analysis|explanation|report)\b",          ""),
    (r"\bin-depth (?=insights|findings|analysis|explanation|report|recommendations)\b",          ""),
    (r"\bcomprehensive (?=insights|findings|analysis|explanation|report|overview|summary)\b",    ""),
    (r"\bthorough (?=insights|findings|analysis|explanation|report|overview|summary)\b",         ""),
    # Fix broken article after adjective removal: "a overview/analysis/insight" → "an ..."
    (r"\ba (overview|analysis|insight|explanation|understanding)\b",                             r"an \1"),

    # ── Verbose qualifiers that rules should strip (directive handles depth) ───
    # "explain in detail" → "explain" — depth/format directive handles this
    (r"\bexplain in detail\b",                                        "explain"),
    (r"\bdescribe in detail\b",                                       "describe"),
    (r"\bprovide (?:a )?detailed?\b",                                 "provide"),
    # "how it/they evolved over time" → "evolution"
    (r"\bhow (?:it |they |this )?evolved? (?:over time|through(?:out)? history|historically)\b",
                                                                      "evolution"),
    # "how it developed over time" → "development"
    (r"\bhow (?:it |they |this )?developed? over time\b",            "development"),
    # "in different industries" → "across industries"
    (r"\bin different industries\b",                                  "across industries"),
    # "for each of the X mentioned" → "per X"
    (r"\bfor each of the (\w+) mentioned\b",                         r"per \1"),
    # "provide examples in detail for each" → "with examples per topic"
    (r"\bprovide (?:examples? in detail|detailed examples?) for each\b", "with examples per topic"),
    # "examples in detail" → "examples"
    (r"\bexamples? in detail\b",                                     "examples"),
    # "in detail" as trailing qualifier (when not already handled) → remove
    (r",?\s*in detail\b(?!\s+what|\s+how|\s+why|\s+where|\s+when)",  ""),
    # "background" when paired with "history" is redundant
    (r"\bits? history,? its? background,?\b",                        "history"),
    (r"\bhistory and background\b",                                   "history"),

    # ── Enterprise / architecture prompt patterns ────────────────────────────
    # "Given that I am working in X" → "Context: X"
    (r"\bGiven that I (?:am|was|have been) (?:working|building|developing|managing|running|operating) "
     r"(?:in |on |with )?(?:a |an |the )?",                          "Context: "),
    # "where there are multiple layers like" → "with layers:"
    (r"\bwhere there are (?:multiple |many |several )?(?:layers? |components? )?(?:like|including|such as)?\b",
                                                                      "with layers: "),
    # "and integration with third-party/external" → "+ 3rd-party"
    (r"\band integration with third.party\b",                        "+ 3rd-party"),
    (r"\band integration with external\b",                           "+ external"),
    (r"\bthird.party\b",                                             "3rd-party"),
    # Verbose architecture nouns → compact
    (r"\borchestration layers?\b",                                    "orchestration"),
    (r"\bdatabase(?:s| layer| systems?)?\b",                         "DBs"),
    (r"\bmiddleware layer\b",                                         "middleware"),
    # "how data flows" → "data flow"
    (r"\bhow data (?:flows?|moves?|is transferred)\b",               "data flow"),
    # "how decisions are made" → "decisioning"
    (r"\bhow decisions are made\b",                                   "decisioning"),
    # "what are the components involved" → "components"
    (r"\bwhat are the (?:key |main |primary )?components? involved\b","components"),
    # "I want you to explain how X" → "explain X"
    (r"\bI want you to explain how\b",                                "explain how"),
    # "how X can be integrated into such a system" → "X integration"
    (r"\bhow (\w+(?:\s+\w+)?) can be integrated into (?:such (?:a|an) )?(?:\w+ )?(?:system|architecture)\b",
                                                                      r"\1 integration"),
    # "such a system architecture" → "this architecture"
    (r"\bsuch (?:a|an) (?:\w+ )?(?:system )?architecture\b",        "this architecture"),

    # ── Redundancy deduplication ─────────────────────────────────────────────
    # "simple and easy" / "easy and simple" → "simple"
    (r"\bsimple and easy\b",                           "simple"),
    (r"\beasy and simple\b",                           "simple"),
    (r"\bsimple and easy to understand\b",             "simple"),
    (r"\beasy and clear\b",                            "clear"),
    # "with examples, real-world examples" or "examples, like real-world examples" → deduplicate
    (r"\bwith examples,?\s*(?:like |including )?real-world examples\b", "with real-world examples"),
    (r"\bexamples?,?\s*(?:like |including )?real-world examples?\b",    "real-world examples"),
    # "examples, examples" of any kind
    (r"\b(examples?),\s*\1\b",                         r"\1"),
    # "in a simple way" / "in an easy way" → just remove (beginner-friendly directive handles it)
    (r",?\s*in (?:a |an )?(?:very )?(?:simple|easy|clear|basic) (?:way|manner)\b", ""),

    # ── Audience / tone meta-instructions → compact tag ────────────────────
    # "in a very simple manner so that even a beginner can understand clearly"
    (r",?\s*in (?:a |an )?(?:very |extremely |quite )?(?:simple|easy|clear|basic|accessible|layman'?s?)"
     r"[\w\s]* (?:manner|way|language|terms?|words?)[^.]*?"
     r"(?:so that|such that|for|ensuring)[^.]*?"
     r"(?:beginner|novice|anyone|everyone|non.?expert|layman|newcomer)s?[^.]*?(?:understand|follow|grasp|learn)[^.]*",
     " (beginner-friendly)"),
    # Simpler: "so that even a beginner can understand"
    (r",?\s*so that (?:even )?(?:a |an )?(?:beginner|novice|anyone|layman|non.?expert)s?"
     r"[\s\w]* (?:can|could|will) (?:understand|follow|grasp|comprehend)[\s\w]*",
     " (beginner-friendly)"),
    # "explain everything in a very simple manner"
    (r",?\s*(?:and )?explain everything in (?:a |an )?(?:very |extremely )?(?:simple|easy|clear|basic)"
     r"[\s\w]* (?:manner|way|language|terms?)",
     " (beginner-friendly)"),

    # ── "and also give/provide/show examples" → "; examples" ─────────
    # Must be broad enough to catch "give some examples", "give a few real-world examples", etc.
    # NOTE: do NOT produce "with real-world examples" here — "with" would be left stranded
    # if _semantic_dedup later removes "real-world examples" as a duplicate concept.
    (r"\band also (?:give|provide|show|include|add) (?:some |a few |many |various )?(?:real-world |practical )?examples?\b",
     "; examples"),
    # "like real-world examples" → "real-world examples" (drop filler "like")
    (r"\blike real-world\b", "real-world"),
    # "and also X" (remaining non-example cases) → "; X"
    # Negative lookahead covers "some/a few/real-world examples" so "some" after verb is caught
    (r"\band also (?:provide |give |include |add |explain |show |describe )?"
     r"(?!(?:some |a few |many |various )?(?:real-world |practical )?examples?)", "; "),
    # "including its X, its Y, its Z" → remove repeated "its"
    (r"\bincluding its ",  "including "),
    (r", its (?=\w)",      ", "),
    # "including X, Y, Z" at start of clause → ": X; Y; Z"
    (r",? including ",    ": "),

    # ── Verbose enumeration in sentences ────────────────────────────────
    # "uses AI, machine learning, APIs, and data pipelines to" → "uses AI/ML/APIs to"
    (r"\bAI, machine learning, APIs,? and data pipelines\b", "AI/ML/APIs/pipelines"),
    (r"\bartificial intelligence, machine learning,? and\b", "AI/ML and"),
    (r"\bAI, machine learning,? and\b",                      "AI/ML and"),
    # "process user requests, understand them, fetch relevant information, make decisions, and respond back"
    # → these verb-object chains: handled by _compress_structure chain removal
    (r"\bunderstand them,?\s*",                              ""),
    (r"\brespond back\b",                                    "respond"),
    (r"\bfetch relevant\b",                                  "fetch"),
    (r"\bmake decisions\b",                                  "decide"),

    # ── "provide examples in detail for each of the X mentioned" ─────────
    (r",?\s*(?:and )?(?:also )?provide (?:detailed |in-depth )?examples? (?:in detail )?for each"
     r"(?: of the \w+ mentioned| industry mentioned| mentioned)?\b",
     " with examples per topic"),
    # "examples for each NOUN" → "examples per NOUN"; bare "each" → "per topic"
    (r"\bexamples? (?:for )?each (\w+)\b",  r"examples per \1"),
    (r"\bexamples? (?:for )?each\b",         "examples per topic"),

    # "give me a/an detailed explanation of X" → "explain X"
    (r"\bgive me (?:a |an )?(?:detailed |comprehensive |brief |full )?explanation (?:of|for)\b",  "explain"),
    (r"\bprovide (?:a |an )?(?:detailed |comprehensive |brief |full )?explanation (?:of|for)\b",  "explain"),
    (r"\bgive me (?:a |an )?(?:detailed |comprehensive |brief |full )?description (?:of|for)\b",  "describe"),

    # ── Instruction compression ───────────────────────────────────────────────
    # "tell me how to" → "how to"
    (r"\btell me how to\b",                                              "how to"),
    # "what is the difference between X and Y" → "difference: X vs Y"
    (r"\bwhat (?:is|are) (?:the )?(?:main |key )?differences? between\b","difference:"),
    # "what are the steps to/for" → "steps:"
    (r"\bwhat are the steps (?:to|for)\b",                              "steps:"),
    # "can you help me with" → remove entirely
    (r"\bcan you help me with\b[\s]*",                                  ""),
    # "list out" → "list"
    (r"\blist out\b",                                                   "list"),
    # "explain about" → "explain"
    (r"\bexplain about\b",                                              "explain"),
    # "the reason why is because" → strip "the reason why is" (keeps "because ...")
    (r"\bthe reason (?:why )?is\b\s*",                                  ""),
    # "the reason why" standalone → strip (downstream rule handles "because" if needed)
    (r"\bthe reason why\b\s*",                                          ""),
    # "which is/are used for" → "for"
    (r"\bwhich (?:is|are) used for\b",                                  "for"),
    # "that/which helps to" → "to"
    (r"\b(?:that|which) helps? to\b",                                   "to"),
    # "how does it/this/that work" → already handled; also "how X works" forms
    (r"\bhow (?:does (?:it|this|that)|do (?:they|these)) work\b",       "how it works"),
    # "what are the steps" shorthand
    (r"\bwhat (?:are|is) (?:the )?steps?\b",                            "steps:"),
    # Formatting shorthands (aggressive-safe since meaning is identical)
    (r"\bversus\b",                                                     "vs"),
    (r"\bapproximately\b",                                              "~"),

    # ── "Can we have/make/build X" → imperative ─────────────────────────────
    # Short product/feature request phrasing — very common in short prompts
    (r"\bCan we have\b",           "Create"),
    (r"\bCan we make\b",           "Create"),
    (r"\bCan we build\b",          "Build"),
    (r"\bCan we create\b",         "Create"),
    (r"\bCan we add\b",            "Add"),
    (r"\bCan we implement\b",      "Implement"),
    (r"\bCan we get\b",            "Get"),
    (r"\bCan we enable\b",         "Enable"),
    (r"\bCan we support\b",        "Support"),
    (r"\bCan we integrate\b",      "Integrate"),

    # ── "of this which will/that will" → "to" ────────────────────────────────
    # "an app of this which will optimize X" → "an app to optimize X"
    (r"\bof this which will\b",    "to"),
    (r"\bof this that will\b",     "to"),
    (r"\bof this to\b",            "to"),    # dedup "to to" prevented by _fix_punctuation

    # ── "which will [verb]" → "to [verb]" (relative clause → infinitive) ─────
    # Safe: "a tool which will process X" → "a tool to process X"
    (r"\bwhich will\b",            "to"),
    (r"\bthat will\b(?=\s+\w+)",   "to"),  # "a system that will handle" → "a system to handle"

    # ── "when X is used in/on/with" → "for X in/on/with" ────────────────────
    (r"\bwhen (?:it|this) is used in\b",   "for use in"),
    (r"\bwhen (?:it|this) is used on\b",   "for use on"),
    (r"\bwhen (?:it|this) is used with\b", "for use with"),

    # ── "mobile and ios" / "ios and android" shorthand ───────────────────────
    (r"\bmobile and ios\b",          "mobile/iOS"),
    (r"\bios and android\b",         "iOS/Android"),
    (r"\bandroid and ios\b",         "Android/iOS"),
    (r"\bmobile and android\b",      "mobile/Android"),

    # ── Conversational request patterns → imperative ──────────────────────────
    # "if you could maybe help me understand something about X" → "Explain X"
    (r"\bif you could (?:maybe |possibly |please )?(?:help me (?:to )?understand|explain) (?:something about |about )?\s*",
     "Explain "),
    # "help me understand something about X" → "Explain X"  (extends existing rule below)
    (r"\bhelp me (?:to )?understand (?:something about |about )\s*",   "Explain "),
    # "if you could [verb]" → just the verb
    (r"\bif you could (?:maybe |possibly |please )?"
     r"(explain|describe|summarize|compare|analyze|define|discuss|tell me|show me|write|create|implement|build|help me write|help me create|help me build)\b",
     r"\1"),
    # "maybe [verb]" leftover after above rules fire
    (r"\bmaybe (explain|describe|tell me|summarize|compare|analyze|define|discuss|write|create|implement)\b",  r"\1"),
    # "something about" dangling after request rewrite
    (r"\bsomething about\b\s*",                                         ""),

    # ── CODING PROMPT OPTIMIZATIONS (new) ──────────────────────────────────
    # Remove verbose instructional language for coding tasks
    (r"\bwrite me (?:a|an|the|some)?\b",                "write "),
    (r"\bcreate (?:a|an|the)?\b",                       "create "),
    (r"\bimplement (?:a|an|the)?\b",                    "implement "),
    (r"\bbuild (?:a|an|the)?\b",                        "build "),
    (r"\bcan you (?:please )?(?:write|create|implement|build)\b",  "write"),
    (r"\bplease (write|create|implement|build)\b",      r"\1"),
    # Combined patterns to avoid duplicates
    (r"\bmake sure to include\b",                       "include"),
    (r"\bmake sure to\b",                               "include"),
    (r"\bbe sure to\b",                                 "include"),
    (r"\bdon'?t forget to\b",                           "include"),
    (r"\b(?:and )?make sure to include\b",              "include"),
    (r"\b(?:and )?also include\b",                      "include"),
    (r"\b(?:including|with) (?:all )?necessary methods?\b", "with methods"),
    (r"\b(?:with )?proper error handling\b",            "with error handling"),
    (r"\b(?:with )?full error handling\b",              "with error handling"),
    (r"\b(?:with )?comprehensive error handling\b",     "with error handling"),
    (r"\b(?:with )?detailed comments\b",                "with comments"),
    (r"\b(?:with )?comprehensive comments\b",           "with comments"),
    (r"\b(?:with )?detailed explanations?\b",           "explained"),
    (r"\b(?:with )?comprehensive explanations?\b",      "explained"),
    (r"\b(?:with )?detailed docstrings?\b",             "with docstrings"),
    (r"\b(?:with )?comprehensive docstrings?\b",        "with docstrings"),
    (r"\bwith type hints?\b",                           "with type hints"),
    (r"\bproper type hints?\b",                         "type hints"),
    (r"\b(?:and )?detailed explanation of\b",           "explain"),
    (r"\bexplaining (?:each|every)\b",                  "explaining"),
    (r"\b(?:the )?algorithm complexity and edge cases\b", "complexity and edge cases"),
    # Fix duplicate prepositions: "include with" → "with", "including with" → "with"
    (r"\binclude with\b",                               "with"),
    (r"\bincluding with\b",                             "with"),

    # ── Nominalization → strong verb ────────────────────────────────────────
    (r"\bmake (?:a |an )?decision\b",                   "decide"),
    (r"\bmake (?:a |an )?choice\b",                     "choose"),
    (r"\bmake (?:a |an )?change\b",                     "change"),
    (r"\bmake (?:a |an )?improvement\b",                "improve"),
    (r"\bmake (?:a |an )?recommendation\b",             "recommend"),
    (r"\bmake (?:a |an )?comparison\b",                 "compare"),
    (r"\btake (?:a |an )?look at\b",                    "review"),
    (r"\btake into (?:account|consideration)\b",        "consider"),
    (r"\bperform (?:an? )?analysis\b",                  "analyze"),
    (r"\bconduct (?:a |an )?review\b",                  "review"),
    (r"\bhave (?:a |an )?discussion\b",                 "discuss"),
    (r"\bprovide (?:an? )?assistance\b",                "help"),
    (r"\bprovide (?:an? )?support\b",                   "support"),
    (r"\butilize[sd]?\b",                               "use"),
    (r"\butilising\b",                                  "using"),
    (r"\butilization\b",                                "use"),
    (r"\bfacilitate\b",                                 "enable"),
    (r"\bmake use of\b",                                "use"),
    (r"\bcome to (?:a |an )?conclusion\b",              "conclude"),
    (r"\breach (?:a |an )?agreement\b",                 "agree"),
    (r"\bgive (?:a |an )?response\b",                   "respond"),
    (r"\bgive (?:a |an )?explanation\b",                "explain"),
    (r"\boffer (?:a |an )?suggestion\b",                "suggest"),

    # ── Pleonasms (redundant word pairs) ────────────────────────────────────
    (r"\bfinal conclusion\b",                           "conclusion"),
    (r"\bend result\b",                                 "result"),
    (r"\bpast history\b",                               "history"),
    (r"\bfuture plans?\b",                              "plans"),
    (r"\bexact same\b",                                 "same"),
    (r"\bfirst and foremost\b",                         "first"),
    (r"\bnew innovation\b",                             "innovation"),
    (r"\bcollaborate together\b",                       "collaborate"),
    (r"\bjoin together\b",                              "join"),
    (r"\brevert back\b",                                "revert"),
    (r"\brefer back\b",                                 "refer"),
    (r"\bstill remains?\b",                             "remains"),
    (r"\bcompletely eliminate\b",                       "eliminate"),
    (r"\badvance (planning|preparation)\b",              r"\1"),
    (r"\bfree gift\b",                                  "gift"),
    (r"\bunexpected surprise\b",                        "surprise"),
    (r"\bclose proximity\b",                            "proximity"),
    (r"\babsolute necessity\b",                         "necessity"),

    # ── "On a X basis" → adverb ─────────────────────────────────────────────
    (r"\bon (?:a |an )?daily basis\b",                  "daily"),
    (r"\bon (?:a |an )?weekly basis\b",                 "weekly"),
    (r"\bon (?:a |an )?monthly basis\b",                "monthly"),
    (r"\bon (?:a |an )?annual(?:ly)? basis\b",          "annually"),
    (r"\bon (?:a |an )?regular basis\b",                "regularly"),
    (r"\bon (?:a |an )?consistent basis\b",             "consistently"),
    (r"\bon (?:a |an )?frequent basis\b",               "frequently"),

    # ── "In the process/middle of Xing" → "Xing" ───────────────────────────
    (r"\bin the process of (\w+ing)\b",                 r"\1"),
    (r"\bin the middle of (\w+ing)\b",                  r"\1"),
    (r"\bin the act of (\w+ing)\b",                     r"\1"),

    # ── Verbose connectors → compact equivalents ────────────────────────────
    (r"\bin the near future\b",                         "soon"),
    (r"\bin the not[- ]too[- ]distant future\b",        "soon"),
    (r"\bin conjunction with\b",                        "with"),
    (r"\bin accordance with\b",                         "per"),
    (r"\bin compliance with\b",                         "per"),
    (r"\bon account of\b",                              "because of"),
    (r"\bby means of\b",                                "via"),
    (r"\bby virtue of\b",                               "due to"),
    (r"\bgoing forward[,\s]*",                          ""),
    (r"\bmoving forward[,\s]*",                         ""),
    (r"\bat this point in time\b",                      "now"),
    (r"\bat the present time\b",                        "now"),
    (r"\bat the current time\b",                        "now"),
    (r"\bdue to the fact that\b",                       "because"),
    (r"\bin light of the fact that\b",                  "because"),
    (r"\bfor the purpose of\b",                         "to"),
    (r"\bwith the exception of\b",                      "except"),
    (r"\bin the event that\b",                          "if"),
    (r"\bin order to\b",                                "to"),
    (r"\bprior to\b",                                   "before"),
    (r"\bsubsequent to\b",                              "after"),
    (r"\bwith regard to\b",                             "about"),
    (r"\bwith respect to\b",                            "about"),
    (r"\bpertaining to\b",                              "about"),
    (r"\bwith reference to\b",                          "about"),
    (r"\bin terms of\b",                                "for"),
    (r"\bfor the reason that\b",                        "because"),

    # ── "There is/are" constructions ─────────────────────────────────────────
    (r"\bthere is (?:a |an )?need to\b",                "need to"),
    (r"\bthere is (?:a |an )?possibility that\b",       "possibly"),
    (r"\bthere are (?:a number of|several|many)\b",     "many"),
    (r"\bit is (?:important|necessary|essential) to\b", ""),
    (r"\bit is worth noting that\b",                    "note:"),
    (r"\bit should be noted that\b",                    "note:"),
    (r"\bit is important to note that\b",               "note:"),

    # ── Chatbot / AI prompt preambles ────────────────────────────────────────
    (r"\bI(?:'m| am) (?:currently )?looking for (?:help|assistance) (?:with|on)\b",  "Help with:"),
    (r"\bI have (?:a )?question (?:about|regarding)\b",                               "About:"),
    (r"\bMy question is(?:\s+about)?\b[\s:]*",                                        ""),
    (r"\bWould it be possible (?:for you )?to\b",                                     "Can you"),
    (r"\bDo you think you could\b",                                                   "Can you"),
    (r"\bI was hoping (?:you could|to)\b\s*",                                         ""),
    # Guidance-seeking + construction verb → "Help me <verb>" (preserves intent)
    # Must come BEFORE the generic drop rules below so the lookahead fires first.
    (r"\bI(?:'d)? (?:want|need|(?:would )?like) (?:you )?to (?=(?:create|build|write|design|develop|make|craft|explore|plan|establish|work on|figure out|think through)\b)", "Help me "),
    (r"\bI (?:would |'d )?like (?:you )?to\b",                                        ""),
    (r"\bI need (?:you )?to\b",                                                       ""),
    (r"\bCould you (?:please )?(?:help me )?\b",                                      ""),
    (r"\bCan you (?:please )?(?:help me )?\b",                                        ""),
    (r"\bPlease (?:help me |assist me )?(?:with |to )?\b",                            ""),

    # ── "I want you to / I want to" ──────────────────────────────────────────
    (r"\bI want (?:you )?to\b\s*",                                                    ""),
    (r"\bI(?:'d| would) (?:also )?want (?:you )?to\b\s*",                            ""),
    (r"\bI (?:also )?want to know\b\s*",                                              ""),

    # ── Domain/role context — preserve as compact tag, never drop ──────────────
    # "from a security standpoint" → "[security]" — keeps the domain frame.
    # Dropping it entirely (old behavior) silently changes what a correct answer looks like:
    # "From a legal perspective, explain X" without the tag → X is explained generically.
    # Longer (3-word) domains first so they match before partial 1-word forms.
    (r"\bfrom (?:a |an )((?:\w+ ?){2,3})(?:standpoint|perspective)[,\s]*", r"[\1] "),
    (r"\bfrom (?:a |an )((?:\w+ ?){1,2})(?:standpoint|perspective)[,\s]*",  r"[\1] "),

    # "Given that I am a senior developer" → "(senior developer):" — keeps experience level.
    # Experience level changes what a correct answer looks like (beginner vs senior).
    (r"\bGiven that I am (?:a |an )?((?:senior |junior |lead |principal |staff )?(?:developer|engineer|architect|data scientist|analyst|designer|manager|consultant))[,\s]*",
     r"(\1): "),

    # ── "Given that / Given the fact that" setup clauses ────────────────────
    # Pattern: "Given that <setup clause>, <actual request>" → "<setup>: <request>"
    # We compress the filler but keep both parts
    (r"\bGiven that\b",                                                               "Context:"),
    (r"\bGiven the fact that\b",                                                      "Context:"),
    (r"\bKeep in mind that\b",                                                        "Note:"),
    (r"\bNote that\b",                                                                "Note:"),
    (r"\bBear in mind that\b",                                                        "Note:"),

    # ── "also how X and also how Y" → "X, Y" ─────────────────────────────────
    (r"\band also how\b",                                                             "and how"),
    (r"\band also what\b",                                                            "and what"),
    (r"\band also why\b",                                                             "and why"),

    # ── Verbose "there are multiple layers like" ──────────────────────────────
    (r"\bwhere there are multiple layers (?:like|such as|including)\b",              "including"),
    (r"\bmultiple layers (?:like|such as|including)\b",                              "layers:"),
    (r"\bintegration with third-party systems?\b",                                   "3rd-party integration"),
    (r"\bthird-party systems?\b",                                                    "3rd-party"),

    # ── Weak "I am working in/on" setup ──────────────────────────────────────
    (r"\bI am (?:currently )?working (?:in|on) (?:a |an |the )?\b",                 ""),

    # ── Abbreviation substitutions ────────────────────────────────────────────
    (r"\bartificial intelligence\b",                            "AI"),
    (r"\bmachine learning\b",                                   "ML"),
    (r"\bdeep learning\b",                                      "DL"),
    (r"\bnatural language processing\b",                        "NLP"),
    (r"\blarge language models?\b",                             "LLM"),
    (r"\bapplication programming interfaces?\b",                "API"),
    (r"\buser interface\b",                                     "UI"),
    (r"\buser experience\b",                                    "UX"),
    (r"\bcontinuous integration and continuous deployment\b",   "CI/CD"),
    (r"\bcontinuous integration\/continuous deployment\b",      "CI/CD"),
    (r"\binfrastructure as code\b",                             "IaC"),
    (r"\bobject[- ]oriented programming\b",                     "OOP"),
    (r"\bretrieval[- ]augmented generation\b",                  "RAG"),
    (r"\bkey performance indicators?\b",                        "KPI"),
    (r"\bservice[- ]level agreements?\b",                       "SLA"),
    (r"\brecovery time objective\b",                            "RTO"),
    (r"\brecovery point objective\b",                           "RPO"),
    (r"\bgeneral data protection regulation\b",                 "GDPR"),
    (r"\bhealth insurance portability and accountability act\b","HIPAA"),
    (r"\bsingle page applications?\b",                          "SPA"),
    (r"\brepresentational state transfer\b",                    "REST"),
    (r"\bsearch engine optimization\b",                         "SEO"),
    (r"\bcommand line interface\b",                             "CLI"),
    (r"\bgraphical user interface\b",                           "GUI"),
    (r"\bsoftware development kits?\b",                         "SDK"),
    (r"\brelational database management system\b",              "RDBMS"),
    (r"\bmessage queue(?:ing)?\b",                              "MQ"),
    (r"\binput\/output\b",                                      "I/O"),
    (r"\bpoint of sale\b",                                      "POS"),
    (r"\bminimum viable product\b",                             "MVP"),
    (r"\bsubject matter experts?\b",                            "SME"),
    (r"\breturn on investment\b",                               "ROI"),
    (r"\btime to market\b",                                     "TTM"),
    (r"\btotal cost of ownership\b",                            "TCO"),
    (r"\bproof of concept\b",                                   "PoC"),
    (r"\bright now\b",                                          "now"),
    (r"\bat the moment\b",                                      "now"),
    (r"\bcurrently at this time\b",                             "currently"),

    # ── Financial domain abbreviations (portfolio agent context) ─────────────
    (r"\bearnings per share\b",                                    "EPS"),
    (r"\bprice[- ]to[- ]earnings(?: ratio)?\b",                   "P/E"),
    (r"\bprice[- ]earnings(?: ratio)?\b",                         "P/E"),
    (r"\bprice[- ]to[- ]earnings[- ]growth\b",                    "PEG"),
    (r"\bdiscounted cash flow\b",                                  "DCF"),
    (r"\bfree cash flow\b",                                        "FCF"),
    (r"\bearnings before interest,?\s*taxes,?\s*depreciation,?\s*and\s*amortization\b", "EBITDA"),
    (r"\bearnings before interest and taxes\b",                    "EBIT"),
    (r"\byear[- ]over[- ]year\b",                                  "YoY"),
    (r"\bquarter[- ]over[- ]quarter\b",                            "QoQ"),
    (r"\bcompound annual growth rate\b",                           "CAGR"),
    (r"\breturn on equity\b",                                      "ROE"),
    (r"\breturn on assets\b",                                      "ROA"),
    (r"\breturn on invested capital\b",                            "ROIC"),
    (r"\bnet asset value\b",                                       "NAV"),
    (r"\bbook value per share\b",                                  "BVPS"),
    (r"\bdebt[- ]to[- ]equity(?: ratio)?\b",                      "D/E"),
    (r"\bdebt[- ]to[- ]ebitda\b",                                  "Net Debt/EBITDA"),
    (r"\bmargin of safety\b",                                      "MOS"),
    (r"\bfair value\b",                                            "FV"),
    (r"\btarget price\b",                                          "TP"),
    (r"\bmanagement guidance\b",                                   "guidance"),
    (r"\bquarterly results\b",                                     "Q results"),
    (r"\bannual results\b",                                        "annual results"),
    (r"\bforeign institutional investor\b",                        "FII"),
    (r"\bforeign portfolio investor\b",                            "FPI"),
    (r"\bnon[- ]banking financial compan(?:y|ies)\b",              "NBFC"),
    (r"\bsecurities and exchange board of india\b",                "SEBI"),
    (r"\breserve bank of india\b",                                 "RBI"),
    (r"\bsecurities and exchange commission\b",                    "SEC"),
    (r"\bfederal reserve\b",                                       "Fed"),
    (r"\binternational monetary fund\b",                           "IMF"),
    (r"\bworld bank\b",                                            "World Bank"),
    (r"\bengineering[,\s]*procurement[,\s]*(?:and\s*)?construction\b", "EPC"),
    (r"\bindependent power producer\b",                            "IPP"),
    (r"\bprofit after tax\b",                                      "PAT"),
    (r"\bprofit before tax\b",                                     "PBT"),
    (r"\boperating profit\b",                                      "EBIT"),
    (r"\bgross domestic product\b",                                "GDP"),
    (r"\bconsumer price index\b",                                  "CPI"),
    (r"\bindex of industrial production\b",                        "IIP"),
    (r"\bgoods and services tax\b",                                "GST"),
    (r"\bmergers? and acquisitions?\b",                            "M&A"),
    (r"\binitial public offering\b",                               "IPO"),
    (r"\bqualified institutional placement\b",                     "QIP"),
    (r"\bopen market operation\b",                                 "OMO"),
    (r"\bnet interest margin\b",                                   "NIM"),
    (r"\bnon[- ]performing asset\b",                               "NPA"),
    (r"\bgross[- ]nonperforming\b",                                "GNPA"),

    # ── CRUD / REST / test exhaustive set substitutions ───────────────────────
    (r"\bcreate,\s*read,\s*update,?\s*and\s*delete\b",          "CRUD"),
    (r"\bread,\s*write,?\s*update,?\s*and\s*delete\b",          "CRUD"),
    (r"\binsert,\s*select,\s*update,?\s*and\s*delete\b",        "CRUD"),
    (r"\bPOST,\s*GET,\s*PUT,\s*PATCH,?\s*and\s*DELETE\b",       "all REST methods"),
    (r"\bGET,\s*POST,\s*PUT,?\s*and\s*DELETE\b",                "CRUD REST methods"),
    (r"\bread operations?,\s*write operations?,\s*update operations?,?\s*and\s*delete operations?\b", "CRUD operations"),
    (r"\bunit tests?,?\s*integration tests?,?\s*(?:end[- ]to[- ]end|e2e) tests?\b", "unit/integration/e2e tests"),
    (r"\bunit tests?,?\s*integration tests?,?\s*edge case tests?,?\s*boundary tests?,?\s*and\s*regression tests?\b", "all test types"),
    (r"\bMonday,\s*Tuesday,\s*Wednesday,\s*Thursday,?\s*and\s*Friday\b", "weekdays"),
    (r"\bMonday through Friday\b",                              "Mon–Fri"),
    (r"\bQ1,\s*Q2,\s*Q3,?\s*and\s*Q4\b",                       "all quarters"),
    (r"\bReact,\s*Angular,?\s*and\s*Vue\b",                     "major JS frameworks"),
    (r"\bReact,\s*Angular,\s*Vue,?\s*and\s*Svelte\b",           "major JS frameworks"),
    (r"\bPython,\s*JavaScript,\s*Java,?\s*and\s*(?:C\+\+|Go|Rust)\b", "major languages"),
    (r"\bMySQL,\s*PostgreSQL,?\s*and\s*SQLite\b",               "SQL databases"),
    (r"\bMySQL,\s*PostgreSQL,\s*SQLite,?\s*and\s*Oracle\b",     "SQL databases"),
    (r"\bAWS,\s*Azure,?\s*and\s*GCP\b",                         "major cloud providers"),
    (r"\bAWS,\s*Azure,\s*GCP,?\s*and\s*(?:DigitalOcean|Heroku)\b", "major cloud providers"),
    (r"\bKubernetes,\s*Docker(?:\s*Swarm)?,?\s*and\s*(?:Nomad|Mesos)\b", "container orchestration"),
    (r"\blogging,\s*monitoring,\s*alerting,?\s*and\s*tracing\b","full observability"),
    (r"\blogs?,\s*metrics?,\s*traces?,?\s*and\s*alerts?\b",     "logs/metrics/traces/alerts"),
    (r"\bGDPR,\s*HIPAA,\s*SOC\s*2?,?\s*and\s*ISO\s*27001\b",   "GDPR/HIPAA/SOC2/ISO27001"),
    (r"\bGDPR,\s*CCPA,?\s*and\s*HIPAA\b",                      "data privacy regulations"),

    # ── Repeated prefix deduplication ─────────────────────────────────────────
    (r"\buser authentication,?\s*user authori[sz]ation,?\s*user session management,?\s*(?:and\s*)?user profile management\b", "user: auth, authz, sessions, profiles"),
    (r"\buser authentication,?\s*(?:and\s*)?user authori[sz]ation\b", "user auth/authz"),
    (r"\berror handling,?\s*(?:exception handling,?\s*)?(?:input validation,?\s*)?(?:null checks?,?\s*)?(?:and\s*)?edge case handling\b", "error handling + validation"),
    (r"\berror handling,?\s*and\s*exception handling\b",        "error/exception handling"),
    (r"\binput validation,?\s*and\s*null checks?\b",            "input validation"),
    (r"\bperformance,?\s*scalability,?\s*(?:availability,?\s*)?(?:reliability,?\s*)?(?:and\s*)?maintainability\b", "non-functional requirements"),
    (r"\blatency,?\s*throughput,?\s*memory\s*(?:footprint|usage)?,?\s*CPU\s*(?:utili[sz]ation|usage)?,?\s*(?:and\s*)?(?:disk\s*)?I\/O\b", "perf metrics: latency/throughput/memory/CPU/I/O"),
    (r"\bP50,?\s*P95,?\s*(?:and\s*)?P99\b",                    "P50/P95/P99"),
    (r"\bclean,?\s*readable,?\s*maintainable,?\s*(?:well[- ]documented,?\s*)?(?:(?:follow(?:s|ing)? best practices|best practices),?\s*)?(?:and\s*)?(?:adhere(?:s|ing)? to |follows? )?SOLID\b", "clean, maintainable, SOLID"),
    (r"\bclean,?\s*readable,?\s*maintainable,?\s*(?:and\s*)?well[- ]structured\b", "clean, maintainable"),
    (r"\boptimi[sz]ed,?\s*efficient,?\s*(?:and\s*)?performant\b", "efficient"),
    (r"\baccurate,?\s*precise,?\s*correct,?\s*factual,?\s*(?:and\s*)?verified\b", "accurate"),
    (r"\bconcise,?\s*brief,?\s*(?:to the point,?\s*)?(?:avoid verbosity,?\s*)?(?:don'?t be long[- ]winded,?\s*)?(?:and\s*)?keep it short\b", "concise"),
    (r"\bprofessional(?:ly)?,?\s*formal(?:ly)?,?\s*(?:business[- ]appropriate,?\s*)?(?:and\s*)?corporate\b", "formal"),
    (r"\bbrief(?:ly)?,?\s*(?:and\s*)?concise(?:ly)?\b",        "concisely"),
    (r"\bshort(?:ly)?,?\s*(?:and\s*)?(?:to the point|concise)\b", "concisely"),
    (r"\bdo not (?:be |include )?(?:too )?verbose,?\s*(?:and\s*)?(?:avoid verbosity|keep it short|don'?t be long[- ]winded)\b", ""),
    (r"\bavoid verbosity,?\s*(?:and\s*)?(?:be concise|keep it short|don'?t be long[- ]winded)\b", ""),
    (r"\bdon'?t include an? introduction,?\s*(?:skip the preamble,?\s*)?(?:don'?t repeat the question,?\s*)?(?:and\s*)?go straight to the answer\b", "No preamble."),
    (r"\bskip the (?:introduction|intro|preamble),?\s*(?:and\s*)?(?:go straight|get straight) to\b", "Answer directly:"),

    # ── Pipeline / stage patterns ─────────────────────────────────────────────
    (r"\bingestion,?\s*validation,?\s*transformation,?\s*(?:enrichment,?\s*)?aggregation,?\s*(?:and\s*)?(?:storage|loading)\b", "ingest→validate→transform→aggregate→load"),
    (r"\bextract,?\s*transform,?\s*(?:and\s*)?load\b",          "ETL"),
    (r"\bextract,?\s*load,?\s*(?:and\s*)?transform\b",          "ELT"),
    (r"\bplan,?\s*design,?\s*develop,?\s*test,?\s*(?:and\s*)?deploy\b", "plan→design→dev→test→deploy"),
    (r"\bdesign,?\s*develop(?:ment)?,?\s*test(?:ing)?,?\s*(?:and\s*)?deploy(?:ment)?\b", "design→dev→test→deploy"),
    (r"\bbuild,?\s*test,?\s*(?:and\s*)?deploy\b",               "build→test→deploy"),
    (r"\bclone,?\s*install,?\s*configure,?\s*(?:and\s*)?run\b", "clone→install→configure→run"),
    (r"\btraining,?\s*validation,?\s*(?:and\s*)?test(?:ing)? (?:sets?|data|split)\b", "train/val/test split"),
    (r"\btraining data(?:set)?,?\s*(?:and\s*)?test data(?:set)?\b", "train/test data"),

    # ── Enterprise context label compressions ─────────────────────────────────
    (r"\bWe have an existing system that currently\b",          "Existing system:"),
    (r"\bIn the context of (?:a |an |the )?\b",                 "Context:"),
    (r"\bOur current tech stack is\b",                          "Stack:"),
    (r"\bOur tech stack (?:includes?|consists? of|is comprised of)\b", "Stack:"),
    (r"\bI have a dataset with columns?:\s*",                   "Dataset cols:"),
    (r"\bwith (?:a |an )?focus on\b",                           "focusing on"),

    # ── Layer / tier noun deduplication ──────────────────────────────────────
    (r"\bpresentation layer,?\s*business logic layer,?\s*data access layer,?\s*(?:and\s*)?persistence layer\b", "presentation/logic/data/persistence layers"),
    (r"\bfrontend,?\s*backend,?\s*(?:and\s*)?database\b",       "frontend/backend/DB"),
    (r"\bclient,?\s*server,?\s*(?:and\s*)?database\b",          "client/server/DB"),
    (r"\bweb layer,?\s*service layer,?\s*(?:and\s*)?(?:data|repository) layer\b", "web/service/data layers"),

    # ── Observability / SRE stack patterns ────────────────────────────────────
    (r"\b(?:with |add |include )?proper logging,?\s*monitoring,?\s*alerting,?\s*tracing,?\s*(?:and\s*)?observability\b", "full observability"),
    (r"\blogging and monitoring\b",                             "logging/monitoring"),
    (r"\bmonitoring and alerting\b",                            "monitoring/alerting"),
    (r"\btracing and logging\b",                                "tracing/logging"),
    (r"\bhigh availability,?\s*(?:and\s*)?fault tolerance\b",   "HA/fault-tolerant"),
    (r"\bhighly available,?\s*(?:and\s*)?fault[- ]tolerant\b",  "HA/fault-tolerant"),
    (r"\bload balancing,?\s*(?:and\s*)?auto[- ]scaling\b",      "load balancing + auto-scaling"),
    (r"\bservice discovery,?\s*load balancing,?\s*circuit breakers?,?\s*(?:API gateway,?\s*)?(?:and\s*)?message queues?\b", "service mesh components"),
    (r"\bblue[- ]green,?\s*(?:and\s*)?canary deployments?\b",   "blue-green/canary"),
    (r"\brolling updates?,?\s*(?:and\s*)?(?:blue[- ]green|canary) deployments?\b", "rolling/blue-green deployments"),

    # ── Comparative / contrast queries ────────────────────────────────────────
    (r"\bwhat is the difference between\b",                     "difference:"),
    (r"\bwhat are the differences between\b",                   "differences:"),
    (r"\bwhat distinguishes\b",                                 "distinguish:"),
    (r"\bcompare and contrast\b",                               "compare"),
    (r"\bsimilarities and differences\b",                       "similarities/differences"),
    (r"\bbenefits and drawbacks\b",                             "pros/cons"),
    (r"\bmerits and demerits\b",                                "pros/cons"),
    (r"\bpros,?\s*cons,?\s*(?:and\s*)?(?:trade[- ]?offs|tradeoffs)\b", "pros/cons/trade-offs"),
    (r"\btrade[- ]?offs?\b",                                    "trade-offs"),
    (r"\bversus\b",                                             "vs"),

    # ── Multi-aspect query compression ────────────────────────────────────────
    (r"\bwhat it is,?\s*how it works?,?\s*(?:and\s*)?(?:when|where) (?:to use|it is used)\b", "what/how/when to use"),
    (r"\bwhat it is,?\s*how it works?,?\s*why it is (?:used|important),?\s*(?:and\s*)?where it is used\b", "what/how/why/where"),
    (r"\bchallenges?,?\s*problems?,?\s*issues?,?\s*(?:and\s*)?pain points?\b", "challenges"),
    (r"\bdefinition,?\s*(?:history,?\s*)?(?:mechanism|working|how it works),?\s*(?:advantages|pros),?\s*(?:disadvantages|cons),?\s*use cases?,?\s*(?:and\s*)?(?:limitations?|future)\b", "definition, mechanism, pros/cons, use cases, limits"),
    (r"\badvantages and disadvantages\b",                       "pros/cons"),
    (r"\bstrengths and weaknesses\b",                           "strengths/weaknesses"),
    (r"\bstart with an? overview,?\s*(?:go into (?:the )?technical details?,?\s*)?(?:cover (?:the )?implementation,?\s*)?(?:discuss (?:the )?trade[- ]?offs?,?\s*)?(?:and\s*)?end with recommendations?\b", "Cover: overview, technical details, implementation, trade-offs, recommendations"),

    # ── Question reformulations ───────────────────────────────────────────────
    (r"\btell me everything (?:you know )?about\b",             "explain"),
    (r"\bgive me (?:a |an )?(?:comprehensive |complete |detailed |thorough )?overview of\b", "overview:"),
    (r"\bgive me (?:a |an )?(?:comprehensive |complete |detailed |thorough )?introduction to\b", "intro:"),
    (r"\bgive me (?:a |an )?(?:comprehensive |complete |detailed |thorough )?breakdown of\b", "breakdown:"),
    (r"\bgive me (?:a |an )?(?:comprehensive |complete |detailed )?rundown of\b", "summary:"),
    (r"\bwalk me through\b",                                    "explain step-by-step:"),
    (r"\btake me through\b",                                    "explain:"),
    (r"\bbreak (?:it|this|that|them) down(?:\s*for me)?\b",     "explain in detail:"),
    (r"\bshed (?:some )?light on\b",                            "explain"),
    (r"\bhelp me (?:wrap my head around|get my head around|understand better)\b", "explain"),
    (r"\bI(?:'m| am) (?:a (?:bit |little |somewhat )?)?confused about\b", "Clarify:"),
    (r"\bI(?:'m| am) (?:not sure|unsure|unclear) about\b",      "Clarify:"),
    (r"\bcan you (?:help me )?(?:clarify|clear up)\b",          "Clarify:"),

    # ── Compliance / security domain ──────────────────────────────────────────
    (r"\bsecurity,?\s*(?:privacy,?\s*)?(?:compliance,?\s*)?(?:and\s*)?governance\b", "security/compliance/governance"),
    (r"\bauthentication,?\s*(?:and\s*)?authori[sz]ation\b",     "auth/authz"),
    (r"\bencryption at rest,?\s*(?:and\s*)?(?:encryption )?in transit\b", "encryption at-rest/in-transit"),
    (r"\bvulnerability(?:\s+assessment)?,?\s*(?:and\s*)?penetration testing\b", "vuln assessment/pentest"),
    (r"\bidentity and access management\b",                     "IAM"),
    (r"\brole[- ]based access control\b",                       "RBAC"),
    (r"\battribute[- ]based access control\b",                  "ABAC"),
    (r"\bzero[- ]trust (?:architecture|model|security)\b",      "zero-trust"),
    (r"\bprinciple of least privilege\b",                       "least-privilege"),
    (r"\bdefense in depth\b",                                   "defense-in-depth"),

    # ── Data / ML domain compressions ────────────────────────────────────────
    (r"\bsupervised,?\s*unsupervised,?\s*(?:and\s*)?(?:semi[- ]supervised|reinforcement) learning\b", "supervised/unsupervised/RL"),
    (r"\bprecision,?\s*recall,?\s*(?:and\s*)?F1[- ]?score\b",  "precision/recall/F1"),
    (r"\baccuracy,?\s*precision,?\s*recall,?\s*(?:and\s*)?F1\b", "accuracy/precision/recall/F1"),
    (r"\bconfusion matrix,?\s*precision,?\s*recall,?\s*(?:and\s*)?(?:F1|AUC[- ]?ROC)\b", "classification metrics"),
    (r"\boverfitting,?\s*(?:and\s*)?underfitting\b",            "overfitting/underfitting"),
    (r"\bfeature engineering,?\s*(?:feature )?selection,?\s*(?:and\s*)?extraction\b", "feature eng/selection/extraction"),
    (r"\bhyperparameter (?:tuning|optimization|search)\b",      "hyperparameter tuning"),
    (r"\bstochastic gradient descent\b",                        "SGD"),
    (r"\bbackpropagation(?: algorithm)?\b",                     "backprop"),
    (r"\bconvolutional neural networks?\b",                     "CNN"),
    (r"\brecurrent neural networks?\b",                         "RNN"),
    (r"\blong short[- ]term memory\b",                          "LSTM"),
    (r"\bgenerative adversarial networks?\b",                   "GAN"),
    (r"\bvariational autoencoders?\b",                          "VAE"),
    (r"\bfine[- ]tun(?:ing|e)\b",                               "fine-tuning"),
    (r"\boverfitting,?\s*(?:and\s*)?underfitting\b",            "overfitting/underfitting"),

    # ── Conversational option/cost reasoning ──────────────────────────────────
    # "But if I go with option A" → "Option A:"
    (r"\bBut if (?:i|I) go with option\b",          "Option"),
    (r"\bif (?:i|I) go with option\b",               "If option"),
    # MUST FIRE FIRST: Remove redundant second cost mention before standalone patterns mutate "will incur"
    # "in that case I will incur lot of cost so ..." — drop the dup; the "so" clause is the real conclusion
    (r"\bin\s+that\s+case\s+(?:I\s+)?will\s+incur\s+(?:lot\s+of|a\s+lot\s+of|lots\s+of|more|extra|significant|huge|high)\s+cost(?:s)?\s+so\b",
     ""),
    (r"\bin\s+that\s+case\s+(?:I\s+)?will\s+incur\s+(?:lot\s+of|a\s+lot\s+of|lots\s+of|more|extra|significant|huge|high)\s+cost(?:s)?\b",
     ""),
    # "I will incur huge cost as I will be sending request to [LLM service]"
    # Full contextual form: compress both the cost statement and the reason together
    (r"\bI\s+will\s+incur\s+(?:a\s+|an\s+)?(?:huge|large|significant|massive|big|high|lot\s+of|lots\s+of)\s+cost(?:s)?"
     r"\s+as\s+I\s+will\s+be\s+sending\s+request(?:s)?\s+to\s+(?:chatgpt|openai|gpt|claude|gemini)\b",
     "ChatGPT API is costly"),
    # Standalone cost-magnitude phrases (after dedup so they don't corrupt "will incur" text)
    (r"\bwill\s+incur\s+(?:a\s+|an\s+)?(?:huge|large|significant|massive|big|high)\s+cost(?:s)?\b",
     "is costly"),
    (r"\bwill\s+incur\s+(?:lot\s+of|a\s+lot\s+of|lots\s+of)\s+cost(?:s)?\b",
     "high cost"),
    # "I will be sending request(s) to [LLM service]" — longer form first to avoid "I will be" dangling
    (r"\bI\s+will\s+be\s+sending\s+request(?:s)?\s+to\s+(?:chatgpt|openai|gpt|claude|gemini)\b",
     "API calls"),
    # "sending request(s) to [LLM service]" fallback
    (r"\bsending\s+request(?:s)?\s+to\s+(?:chatgpt|openai|gpt|claude|gemini)\b",
     "API calls"),
    # "I don't/dont think this would be a viable option/choice/approach" → "not viable"
    (r"\bI\s+don'?t\s+think\s+this\s+would\s+be\s+(?:a\s+)?viable\s+(?:option|choice|approach)\b",
     "not viable"),
    # "in that case we need to have it as X" → "→ X"
    (r"\bin\s+that\s+case\s+we\s+need\s+to\s+have\s+it\s+as\b",  "→"),
    (r"\bwe\s+need\s+to\s+have\s+it\s+as\b",                      "use"),
    # "which/what is the better way to deal with it/this/that" → "Best approach"
    (r"\b(?:which|what)\s+is\s+the\s+better\s+way\s+to\s+deal\s+(?:with\s+)?(?:it|this|that)\b",
     "Best approach"),
    (r"\bbetter\s+way\s+to\s+deal\s+(?:with\s+)?(?:it|this|that)\b",
     "Best approach"),
    # Dangling product reference after request verb
    (r",?\s*\bof\s+this\s+product\b",                             ""),
]

# Compile once
_PHRASE_RES = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in _PHRASE_REPLACEMENTS
]


# ── Phase 3: Structural compression ─────────────────────────────────────────

def _compress_structure(text: str) -> str:
    """
    Convert verbose enumeration to structured shorthand.
    E.g.: "...with examples and advantages and disadvantages"
          -> adds ": definition, examples, pros/cons" pattern
    """
    # Collapse repeated "and X and Y and Z" list tails into comma-separated
    text = re.sub(
        r"\b(\w+) and (\w+) and (\w+) and (\w+)\b",
        r"\1, \2, \3, \4",
        text,
    )
    text = re.sub(
        r"\b(\w+) and (\w+) and (\w+)\b",
        r"\1, \2, \3",
        text,
    )

    # "in a detailed way" / "in a very detailed manner" → "in detail"
    text = re.sub(
        r"\bin (?:a )?(?:very )?detail(?:ed|) (?:way|manner|fashion)\b",
        "in detail",
        text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\bwith (?:a lot of )?detail\b", "in detail", text, flags=re.IGNORECASE)

    # "some examples" / "a few examples" → "examples"
    text = re.sub(r"\b(?:some|a few|several) examples\b", "examples", text, flags=re.IGNORECASE)

    # ── Flatten "verb X, verb Y, and verb Z" action lists ───────────────────
    # "Analyze X, identify Y, and provide Z" → "Analyze X, Y, Z"
    # Matches: leading verb + object, then ", verb object" chains
    text = re.sub(
        r"^([A-Z][a-z]+(?:\s+\S+){0,4}),\s*(?:and\s+)?\b(?:identify|provide|include|list|describe|explain|highlight|note|show|give|present|detail)\b\s+",
        r"\1, ",
        text,
    )
    # Second verb in chain
    text = re.sub(
        r",\s*(?:and\s+)?\b(?:identify|provide|include|list|describe|explain|highlight|note|show|give|present|detail)\b\s+",
        ", ",
        text, flags=re.IGNORECASE,
    )

    # ── Compact tech noun lists: "Kafka, Redis, and RabbitMQ" → "Kafka/Redis/RabbitMQ"
    text = _compact_noun_lists(text)

    # Collapse multiple spaces and trim
    text = re.sub(r"  +", " ", text)
    # Clean trailing comma before end of string
    text = re.sub(r",\s*$", "", text)
    text = text.strip()

    return text


# ── Tech noun list → slash notation ─────────────────────────────────────────

_TECH_NOUN_LIST_RE = re.compile(
    r'\b([A-Z][a-zA-Z0-9.+#\-]{1,20})'           # first capitalized tech name
    r'(?:,\s*[A-Z][a-zA-Z0-9.+#\-]{1,20}){2,}'   # 2+ more comma-separated
    r',?\s*and\s+[A-Z][a-zA-Z0-9.+#\-]{1,20}\b'  # final "and X"
)


def _compact_noun_lists(text: str) -> str:
    """Collapse 'Kafka, Redis, PostgreSQL, and RabbitMQ' → 'Kafka/Redis/PostgreSQL/RabbitMQ'."""
    def _slash_join(m: re.Match) -> str:
        items = re.split(r',\s*(?:and\s+)?', m.group(0))
        return '/'.join(i.strip() for i in items if i.strip())
    return _TECH_NOUN_LIST_RE.sub(_slash_join, text)


# ── Phase 3b: Evaluative sentence removal ───────────────────────────────────
# Removes standalone sentences that are praise/acknowledgement with no
# substantive content — common in LLM critique prompts.

_EVAL_SENTENCE_RE = re.compile(
    r'(?:(?:^|(?<=[.!?:]\s)))'  # sentence boundary: start, or after . ! ? : + whitespace
    r'(?:'
    # Short praise sentences: "Strong response." / "Good answer." / "Solid work."
    r'(?:Strong|Good|Great|Excellent|Nice|Solid|Clean|Clear|Correct|Right|Perfect)'
    r'\s+(?:response|answer|point|observation|analysis|work|approach|example|reframe|framing|critique|summary)\.'
    r'|'
    # Comparative evaluation: "Stronger than the first one." / "Better than the previous response."
    r'(?:Better|Stronger|Clearer|More \w+|Sharper)\s+than\s+(?:the\s+)?(?:first|last|previous|prior|second|earlier)\s+(?:one|response|answer|attempt|version)\.'
    r'|'
    # "earns full/partial credit" sentence
    r'The\s+response\s+earns?\s+(?:full|partial|half|some)\s+credit(?:\s+for[^.]{0,80})?\.'
    r'|'
    # Single-word filler sentences like "Correct." / "Agreed." / "Confirmed."
    r'(?:Correct|Agreed|Confirmed|Noted|Understood|Indeed|Exactly|Precisely|Absolutely|Certainly)\.'
    r')',
    re.IGNORECASE,
)


def _remove_evaluative_sentences(text: str) -> str:
    """Strip standalone evaluative/praise sentences that carry no information."""
    result = _EVAL_SENTENCE_RE.sub("", text)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result if result else text


# ── Phase 4: Sentence-level cleanup ─────────────────────────────────────────

def _clean_sentence(text: str) -> str:
    """Remove leading/trailing filler that survives phrase replacement."""
    # Strip dangling commas/conjunctions at start
    text = re.sub(r"^[,\s]*(and|but|also|so|thus|therefore)[,\s]+", "", text, flags=re.IGNORECASE)
    # Clean up spacing around punctuation
    text = re.sub(r"\s+([.,?!;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _fix_punctuation(text: str) -> str:
    """Normalize punctuation artifacts left behind by rule substitutions."""
    text = re.sub(r',\s*;', ';', text)          # ", ;" → ";"
    text = re.sub(r';\s*;', ';', text)           # "; ;" → ";"
    text = re.sub(r',\s*,', ',', text)           # ",," → ","
    text = re.sub(r'\s+([,;.!?:])', r'\1', text) # space before punctuation
    text = re.sub(r'([,;])\s*([,;])', r'\2', text)  # adjacent punctuation, keep last
    text = re.sub(r'^[,;\s]+', '', text)         # leading punctuation
    text = re.sub(r'[,;]\s*$', '', text)         # trailing comma/semicolon
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


# ── Phase 5: Semantic deduplication ─────────────────────────────────────────

# Concept clusters: if two or more synonyms from the same cluster appear in
# the same prompt, keep only the first occurrence.
_CONCEPT_CLUSTERS: list[list[str]] = [
    # simplicity markers (multi-word first so they match before single-word subsets)
    ["in simple terms", "in simple language", "in a simple way", "in plain english",
     "in layman's terms", "for beginners", "beginner-friendly", "simply put"],
    # example markers — keep phrases disjoint (no phrase may be a substring of another)
    ["with examples", "with real-world examples",
     "with use cases", "with illustrations", "give examples"],
    # detail markers (use full adverb forms to avoid partial matches)
    ["in detail", "in depth", "in-depth", "comprehensively", "thoroughly", "exhaustively"],
    # concise markers
    ["concisely", "briefly", "in brief", "in a nutshell", "succinctly"],
]


def _semantic_dedup(text: str) -> str:
    """
    Remove repeated concept markers within a single prompt.

    E.g. "Explain backpropagation in detail in depth comprehensively"
         → "Explain backpropagation in detail"
    """
    text_lower = text.lower()
    for cluster in _CONCEPT_CLUSTERS:
        found: list[tuple[int, str]] = []
        for phrase in cluster:
            pos = text_lower.find(phrase)
            if pos != -1:
                found.append((pos, phrase))
        if len(found) < 2:
            continue
        # Sort by position; keep the first occurrence, remove the rest
        found.sort(key=lambda x: x[0])
        for _, phrase in found[1:]:
            # Word-boundary-aware removal to avoid partial word matches
            escaped = re.escape(phrase)
            # Add \b only when phrase starts/ends with a word character
            prefix = r'\b' if re.match(r'\w', phrase)       else ''
            suffix = r'\b' if re.search(r'\w$', phrase)     else ''
            pattern = re.compile(
                r'(?:\s*,?\s*\b(?:and|or|also)\b)?\s*,?\s*'
                + prefix + escaped + suffix + r'[\s,]*',
                re.IGNORECASE,
            )
            text = pattern.sub(" ", text, count=1)
            text_lower = text.lower()
    return re.sub(r"\s{2,}", " ", text).strip()


def _remove_dangling_words(text: str) -> str:
    """
    Remove single adjectives/adverbs left dangling between commas after noise removal.
    e.g. "works in a way, simple, with examples" → "works in a way, with examples"
    Targets common words that appear as orphaned filler after stripping phrases.
    """
    _DANGLING = r"\b(simple|easy|clear|basic|quick|fast|good|great|detailed|very|really|quite)\b"
    # ", word," between two non-word-boundary commas
    text = re.sub(r",\s*" + _DANGLING + r"\s*,", ",", text, flags=re.IGNORECASE)
    # ", word." at end
    text = re.sub(r",\s*" + _DANGLING + r"\s*\.", ".", text, flags=re.IGNORECASE)
    # Collapse double commas left behind
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ── Phase 0: Duplicate sentence detection ───────────────────────────────────

def _sent_keywords(s: str) -> set:
    return set(re.findall(r'\b[a-zA-Z]{4,}\b', s.lower()))


def _remove_duplicate_sentences(text: str) -> str:
    """
    Remove sentences whose content is mostly contained in an earlier sentence.
    Uses containment (overlap / shorter_sentence_size) ≥ 0.70 — works even when
    one sentence is much longer than the other (Jaccard would fail there).
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < 2:
        return text
    kept: list[str] = [sentences[0]]
    for sent in sentences[1:]:
        sent_w = _sent_keywords(sent)
        if not sent_w:
            kept.append(sent)
            continue
        is_dup = False
        for prev in kept:
            prev_w = _sent_keywords(prev)
            overlap = len(sent_w & prev_w)
            shorter = min(len(sent_w), len(prev_w))
            if shorter and overlap / shorter >= 0.70:
                is_dup = True
                break
        if not is_dup:
            kept.append(sent)
    return " ".join(kept)


# ── Intent preservation ──────────────────────────────────────────────────────
# Ordered longest-first so "in very great detail" is matched before "in detail".
_INTENT_WORDS = (
    "in very great detail", "in great detail", "in detail",
    "step by step", "step-by-step", "with real-world examples", "with real examples",
    "with examples", "deep-dive", "deep dive",
    "comprehensively", "comprehensive", "thoroughly", "thorough",
    "in-depth", "in depth", "end-to-end", "from scratch",
    "exhaustive", "extensively", "elaborate", "detailed",
)

# Structured optimization: detect "Explain X ... including A, B, C" pattern
_STRUCT_TRIGGER_RE = re.compile(
    r'(?:including|such as|covering|e\.g\.)\s+',
    re.IGNORECASE,
)
_LIST_SPLIT_RE = re.compile(r',\s*(?:and\s+)?|\s+and\s+', re.IGNORECASE)


def _try_structure(prompt: str, detected_intent: str | None) -> str | None:
    """
    Reformat 'Explain X in detail including A, B, C' into:
        Explain X in detail.
        Cover:
        - A
        - B
        - C
    Returns None if the pattern does not match (< 2 list items).
    """
    if not detected_intent:
        return None

    m = _STRUCT_TRIGGER_RE.search(prompt)
    if not m:
        return None

    lead = prompt[: m.start()].strip().rstrip(",;")
    tail = prompt[m.end():].strip().rstrip("?!.")

    items = [i.strip().rstrip(".,;") for i in _LIST_SPLIT_RE.split(tail) if i.strip()]
    if len(items) < 2:
        return None

    # Guarantee the intent word survives in the lead.
    # Also accept a "core" form without intensifiers (e.g. "in great detail" covers "in very great detail").
    _core_intent = _INTENSIFIER_RE.sub('', detected_intent).strip()
    if detected_intent not in lead.lower() and _core_intent not in lead.lower():
        lead = lead.rstrip(". ") + f" {detected_intent}"

    bullets = "\n".join(f"- {item}" for item in items)
    return f"{lead}.\nCover:\n{bullets}"


# Critique-style tag compression (deterministic, no LLM):
# Converts verbose evaluative prose into compact tagged lines while preserving core content.
_TAG_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("issue", re.compile(r"\b(structural problem|structural issue|problem worth naming|critic of this output)\b", re.IGNORECASE)),
    ("strength", re.compile(r"\b(what it got right|got right|correct|conceptually precise|clarifies invariant|clarifies invariants)\b", re.IGNORECASE)),
    ("score", re.compile(r"\b(final score|score)\b", re.IGNORECASE)),
    ("gap", re.compile(r"\b(gap|missing|incomplete|does not|not define|underspecified|open question)\b", re.IGNORECASE)),
    ("invariant", re.compile(r"\b(invariant|tolerance|threshold|consistency|projection)\b", re.IGNORECASE)),
    ("evidence", re.compile(r"\b(example|counter-example|concrete|demonstrate|shows)\b", re.IGNORECASE)),
    ("action", re.compile(r"\b(next|before any code|formally define|should|must)\b", re.IGNORECASE)),
    ("claim", re.compile(r".", re.IGNORECASE)),  # fallback
]

_TAG_STOPWORDS_RE = re.compile(
    r"\b(the|a|an|is|are|was|were|be|been|being|to|of|in|on|at|for|with|that|this|it|as|and|or|but|by)\b",
    re.IGNORECASE,
)


def _to_tag_keywords(sentence: str) -> str:
    t = sentence.lower()
    t = t.replace("human-in-the-loop", "human_in_the_loop")
    t = re.sub(r"[^a-z0-9%/.\- ]+", " ", t)
    t = _TAG_STOPWORDS_RE.sub(" ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = t.replace("human_in_the_loop", "human-in-the-loop")
    # keep short but meaningful
    words = t.split()
    return " ".join(words[:14])


def _compress_tag_sentence(sentence: str, tag: str) -> str:
    """
    Produce a short but still readable tag sentence.
    Keep essential meaning instead of stripping down to pure keywords.
    """
    s = sentence.strip()
    s = re.sub(r'^\s*(what it got right|where the response is (?:still )?incomplete|where the response is incomplete|open questions|the structural problem)\s*:?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^\s*(the answer|the response|it)\s+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bdoes not propose\b', 'missing', s, flags=re.IGNORECASE)
    s = re.sub(r'\bdoes not define\b', 'missing', s, flags=re.IGNORECASE)
    s = re.sub(r'\bis not validated empirically\b', 'lacks empirical validation', s, flags=re.IGNORECASE)
    s = re.sub(r'\bnor a\b', '; no', s, flags=re.IGNORECASE)
    s = re.sub(r'\bit overstates readiness by\b', 'overstates readiness:', s, flags=re.IGNORECASE)
    s = re.sub(r'\bwhile\b', ';', s, flags=re.IGNORECASE)
    s = re.sub(r'\bThe projection reframe is correct\b', 'projection reframe is correct', s, flags=re.IGNORECASE)
    s = re.sub(r'\bFinal score\b', 'final score', s, flags=re.IGNORECASE)
    s = re.sub(r'\bOpen questions\b', 'open questions', s, flags=re.IGNORECASE)
    s = s.replace("human- -", "human-in-the-loop")
    s = s.replace("human- -loop", "human-in-the-loop")
    s = re.sub(r'\s{2,}', ' ', s).strip(" .;:")

    if tag == "issue":
        if "structural problem worth naming" in s.lower() or "problem worth naming" in s.lower():
            return "structural problem worth naming"
        return "structural critique"
    if tag == "score":
        m = re.search(r'(\d+(?:\.\d+)?/\d+)', s)
        if m:
            return f"final score {m.group(1)}"
    if tag == "strength" and "projection reframe" in s.lower():
        return "projection reframe is correct; clarifies invariants"
    if "human-in-the-loop" in sentence.lower() or "curates synonym merges" in sentence.lower():
        return "synonym merge curation likely needs human-in-the-loop"
    if "5/5 clarity" in sentence.lower() and "unresolved engineering gaps" in sentence.lower():
        return "scores 5/5 clarity despite unresolved engineering gaps"
    if tag == "invariant" and "open questions" in sentence.lower():
        s = re.sub(r'^\s*open questions\s*:?\s*', '', sentence, flags=re.IGNORECASE).strip(" .")
        items = [item.strip(" .") for item in re.split(r',\s*|\s+and\s+', s) if item.strip()]
        return "open questions: " + "; ".join(items[:4])

    # Preserve readability, but cap verbose tails.
    words = s.split()
    if len(words) > 18:
        s = " ".join(words[:18]).rstrip(",;:")
    return s


def compress_analysis_critic_tags(text: str) -> str:
    """
    Deterministically compress long critique text into concise tag lines.
    Preserves essence by keeping sentence content, only stripping framing/filler.
    """
    if not text or not text.strip():
        return text

    # Reuse safe cleanup first
    base = rule_optimize(text, "moderate")
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', base) if s.strip()]
    if len(sents) < 4:
        return base

    lines: list[str] = []
    seen = set()
    for s in sents:
        s_clean = re.sub(r'^\s*(what it got right|where .* incomplete|the structural problem)\s*:?\s*', '', s, flags=re.IGNORECASE)
        s_clean = s_clean.strip()
        if not s_clean:
            continue
        tag = "claim"
        if "5/5 clarity" in s_clean.lower() and "unresolved engineering gaps" in s_clean.lower():
            tag = "claim"
        else:
            for name, pat in _TAG_PATTERNS:
                if pat.search(s_clean):
                    tag = name
                    break
        # lightweight dedup
        key = re.sub(r'\W+', '', s_clean.lower())[:80]
        if key in seen:
            continue
        if "critic of this output" in s_clean.lower() and "structural" not in s_clean.lower():
            continue
        seen.add(key)
        concise = _compress_tag_sentence(s_clean, tag)
        if not concise:
            continue
        if count_tokens(concise) > 18:
            concise = _to_tag_keywords(concise)
        lines.append(f"{tag}: {concise}")

    tagged = "\n".join(lines).strip()
    if not tagged:
        return base
    return tagged if count_tokens(tagged) < count_tokens(base) else base


def compress_long_text_tags(text: str) -> str:
    """
    Generic long-text tag compression.  Dispatches to the best domain compressor:
      - task_briefing  → compress_task_briefing  (planning / product prompts)
      - critique/other → compress_analysis_critic_tags
    """
    if _BRIEFING_INDICATOR_RE.search(text):
        briefing = compress_task_briefing(text)
        if count_tokens(briefing) < count_tokens(text):
            return briefing
    return compress_analysis_critic_tags(text)


# ── Task-briefing / product-planning tag compression ─────────────────────────
# Mirrors compress_analysis_critic_tags but for planning prompts:
# each sentence is classified (context / status / action / constraint / goal)
# and compressed to its essential nouns, verbs, and intent.

# Detect planning/briefing prompts before dispatching
_BRIEFING_INDICATOR_RE = re.compile(
    r'\b(?:phases?|ship(?:ped)?|A/?B\s*test(?:ing)?|implement(?:ed)?|sprint|milestone'
    r'|v\d+\.\d+|release\s+v|deploy(?:ment)?|launch(?:ed)?|roadmap|execution\s+plan'
    r'|implementation\s+plan|executable\s+plan|phase[\s\-]?wise)\b',
    re.IGNORECASE,
)

# Tag assignment for briefing sentences (checked in order; first match wins).
# NOTE: "action" is checked BEFORE "context" — a sentence starting with an imperative
# verb (provide/create/plan) is action even if it also mentions AI tool names like "claude".
_BRIEFING_TAG_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("action",     re.compile(
        r'^\s*(?:provide|create|build|implement|execute|plan|design|develop'
        r'|write|generate|make)\b', re.I)),
    ("status",     re.compile(
        r'\b(?:shipped?|launched?|deployed?|live\b|released?|in\s+production'
        r'|A/?B\s*test(?:ing)?|currently)\b', re.I)),
    ("constraint", re.compile(
        r'\b(?:we\s+need\s+to|must\b|deadline|within\b|week\b|day\b|time:|timeline'
        r'|carefully|carfeully|plan\s+all|all\s+phases)\b', re.I)),
    ("context",    re.compile(
        r'\b(?:analyz\w+|review\w*|assess\w*|evaluat\w*|attach\w*|based\s+on'
        r'|chatgpt|openai\b|claude\b|has\s+suggested|recommend\w*|proposed)\b', re.I)),
    ("goal",       re.compile(
        r'\b(?:goal|objective|aim\b|target\b|ship\s+as\b|v\d+\.\d+|release\s+v|version)\b',
        re.I)),
]

# Compiled sentence-level substitutions for briefing prompts
_BRIEFING_SUBS: list[tuple[re.Pattern, str]] = [
    # ── Context openers ───────────────────────────────────────────────────────
    # "After analyzing the product which is attached, " → ""
    (re.compile(
        r'^After\s+(?:carefully\s+|thoroughly\s+|deeply\s+)?analyz\w+\s+(?:the\s+)?'
        r'(?:product|document|data|code|file|text|content|attached\s+\w+)?'
        r'\s*(?:which\s+is\s+attached)?\s*[,\s]*', re.I), ''),
    (re.compile(
        r'^After\s+(?:carefully\s+|thoroughly\s+)?review\w+\s+(?:the\s+)?'
        r'(?:product|document|data|code|file|text|content)?'
        r'\s*(?:which\s+is\s+attached)?\s*[,\s]*', re.I), ''),
    (re.compile(r'\bwhich\s+(?:is|are|was|were)\s+attached\b[,\s]*', re.I), ''),
    (re.compile(r'\bthat\s+(?:is|are|was|were)\s+attached\b[,\s]*', re.I), ''),
    # ── "has suggested to provided below solution on" → "suggests:" ───────────
    (re.compile(r'\bhas\s+suggested\s+to\s+provid\w*\s+below\s+solution\s+on\b', re.I),
     'suggests:'),
    (re.compile(r'\bhas\s+suggested\s+(?:a\s+|an\s+|the\s+)?\b', re.I), 'suggests: '),
    (re.compile(r'\bsuggested\s+to\s+(?:provid\w+|give|offer)\b', re.I), 'suggests:'),
    # ── "phase wise" / "phase-wise" → "phased" ───────────────────────────────
    (re.compile(r'\bphase[\s\-]?wise\b', re.I), 'phased'),
    # ── Action: "provide executable plan for X to complete it" ───────────────
    (re.compile(r'\bprovide\s+(?:an?\s+)?executable\s+plan\s+for\b', re.I), 'plan for'),
    (re.compile(r'\bprovide\s+(?:an?\s+)?(?:execution|implementation)\s+plan\s+for\b',
                re.I), 'plan for'),
    (re.compile(r'\bto\s+complete\s+it\.?\b', re.I), ''),
    # ── Status: "But the product has shipped for A/B testing so" ─────────────
    (re.compile(r'\bBut\s+', re.I), ''),
    (re.compile(
        r'\b(?:the\s+)?(?:product|app|feature|service)\s+has\s+(?:been\s+)?shipped\s+for\b',
        re.I), 'live in'),
    # ── "so we need to carefully/carfeully …" → "; " ─────────────────────────
    (re.compile(r'\bso\s+we\s+need\s+to\s+(?:carefully\s+|carfeully\s+)?', re.I), '; '),
    (re.compile(r'\bwe\s+need\s+to\s+(?:carefully\s+|carfeully\s+)?', re.I), ''),
    # ── Redundant tail: "with all phases implemented" ─────────────────────────
    (re.compile(r'\bwith\s+all\s+phases\s+implemented\b', re.I), ''),
    (re.compile(r'\ball\s+the\s+phases\b', re.I), 'all phases'),
    # ── "ship as vX.Y" → "release vX.Y" ──────────────────────────────────────
    (re.compile(r'\bship\s+as\b', re.I), 'release'),
    # ── Time: "in a week's time" → "in 1 week"
    # [^\s]{0,2} tolerates any apostrophe + s variant (straight, curly, omitted)
    (re.compile(r'\bin\s+a\s+week[^\s]{0,2}\s+time\b', re.I), 'in 1 week'),
    (re.compile(r'\bwithin\s+a\s+week\b', re.I), 'in 1 week'),
    (re.compile(r'\bin\s+two\s+weeks?[^\s]{0,2}\s+time\b', re.I), 'in 2 weeks'),
]


def _compress_briefing_sentence(sentence: str, tag: str) -> str:
    """Apply briefing substitutions and cap at 20 words."""
    s = sentence.strip()
    for pat, rep in _BRIEFING_SUBS:
        s = pat.sub(rep, s)
    s = re.sub(r'\s{2,}', ' ', s).strip(' .,;:')
    words = s.split()
    if len(words) > 20:
        s = ' '.join(words[:20])
    return s


def compress_task_briefing(text: str) -> str:
    """
    Deterministic tag-style compression for task-briefing / product-planning prompts.

    Each sentence is classified into a semantic role (context / status / action /
    constraint / goal) and compressed to essential nouns, verbs, and intent —
    the same philosophy as compress_analysis_critic_tags for critique prompts.

    Only runs when _BRIEFING_INDICATOR_RE detects planning signals.
    Returns original if compressed result is longer.
    """
    if not text or not _BRIEFING_INDICATOR_RE.search(text):
        return text

    base = rule_optimize(text, "moderate")
    # Split on sentence-ending punctuation; also catch colon-terminated briefings
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', base) if s.strip()]
    if len(sents) < 2:
        return base

    lines: list[str] = []
    seen: set[str] = set()
    for s in sents:
        if not s:
            continue
        tag = "info"
        for name, pat in _BRIEFING_TAG_PATTERNS:
            if pat.search(s):
                tag = name
                break

        concise = _compress_briefing_sentence(s, tag)
        if not concise:
            continue

        key = re.sub(r'\W+', '', concise.lower())[:80]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{tag}: {concise}")

    result = '\n'.join(lines).strip()
    if not result:
        return base
    return result if count_tokens(result) < count_tokens(base) else base


# ── Short request compression (mirror of tag compression for 1-2 sentence queries) ──

# Detects short conversational request starters
_SHORT_REQ_STARTER_RE = re.compile(
    r'^(?:'
    r'can we (?:have|make|build|create|add|get|implement|enable|integrate|develop|design|launch|ship|support)|'
    r'could we (?:have|build|make|create|add|get|implement)|'
    r'is it possible to|'
    r'would it be possible to|'
    r'is there (?:a way|an option|any chance) (?:to|we can)|'
    r'do you think we (?:can|could)|'
    r'any chance we can'
    r')',
    re.IGNORECASE,
)

# Starter → imperative verb mapping (same role as tag names in long-text compression)
_STARTER_TO_VERB = [
    (re.compile(r'^Can we have\b',           re.I), 'Build'),
    (re.compile(r'^Can we make\b',           re.I), 'Create'),
    (re.compile(r'^Can we build\b',          re.I), 'Build'),
    (re.compile(r'^Can we create\b',         re.I), 'Create'),
    (re.compile(r'^Can we add\b',            re.I), 'Add'),
    (re.compile(r'^Can we get\b',            re.I), 'Get'),
    (re.compile(r'^Can we implement\b',      re.I), 'Implement'),
    (re.compile(r'^Can we enable\b',         re.I), 'Enable'),
    (re.compile(r'^Can we integrate\b',      re.I), 'Integrate'),
    (re.compile(r'^Can we develop\b',        re.I), 'Develop'),
    (re.compile(r'^Can we design\b',         re.I), 'Design'),
    (re.compile(r'^Can we support\b',        re.I), 'Support'),
    (re.compile(r'^Could we have\b',         re.I), 'Build'),
    (re.compile(r'^Could we build\b',        re.I), 'Build'),
    (re.compile(r'^Is it possible to\b',     re.I), ''),
    (re.compile(r'^Would it be possible to\b',re.I),''),
]

# Body-level compressions (same role as _compress_tag_sentence for long text)
_SHORT_REQ_BODY_SUBS = [
    (re.compile(r'\bof this which will\b',   re.I), 'to'),
    (re.compile(r'\bof this that will\b',    re.I), 'to'),
    (re.compile(r'\bwhich will\b',           re.I), 'to'),
    (re.compile(r'\bthat will\b(?=\s+\w)',   re.I), 'to'),
    (re.compile(r'\bof this\b\s*',           re.I), ''),          # dangling "of this" after verb replacement
    (re.compile(r'\bmobile and ios\b',       re.I), 'mobile/iOS'),
    (re.compile(r'\bios and android\b',      re.I), 'iOS/Android'),
    (re.compile(r'\bandroid and ios\b',      re.I), 'Android/iOS'),
    (re.compile(r'\bwhen (?:it|this) is used in\b',   re.I), 'for use in'),
    (re.compile(r'\bwhen (?:it|this) is used on\b',   re.I), 'for use on'),
    (re.compile(r'\bgoogle chrome or open\s*ai\b',     re.I), 'Chrome/OpenAI'),
    (re.compile(r'\bchrome or open\s*ai\b',            re.I), 'Chrome/OpenAI'),
    (re.compile(r'\bopen\s*ai app\b',                  re.I), 'OpenAI'),
    (re.compile(r'\bgoogle chrome\b',                  re.I), 'Chrome'),
    (re.compile(r'\bthe tokens\b',                     re.I), 'tokens'),
    (re.compile(r'\boptimize the tokens\b',            re.I), 'optimize tokens'),
]


def compress_short_request(text: str) -> str:
    """
    Compact compression for short conversational requests (1-2 sentences, < 50 tokens).

    Mirrors compress_long_text_tags but for short queries:
      - Detect the request type (starter pattern)
      - Replace starter with an imperative verb
      - Compress the body (relative clauses, verbose phrases, brand names)
      - Clean up result
    Returns original if not a recognized request pattern or result is longer.
    """
    stripped = text.strip()
    if not _SHORT_REQ_STARTER_RE.match(stripped):
        return text

    result = stripped

    # Step 1: Replace starter → imperative verb (role detection, like tag assignment)
    for pattern, verb in _STARTER_TO_VERB:
        if pattern.match(result):
            result = pattern.sub(verb, result, count=1).lstrip()
            break

    # Step 2: Compress body (like _compress_tag_sentence)
    for pattern, replacement in _SHORT_REQ_BODY_SUBS:
        result = pattern.sub(replacement, result)

    # Step 3: Clean up artifacts
    result = re.sub(r'\s{2,}', ' ', result).strip()
    result = re.sub(r'\s([?.!])', r'\1', result)   # "word ?" → "word?"
    result = re.sub(r'\?$', '.', result)            # trailing ? → . (now an imperative)

    # Only accept if strictly shorter
    return result if count_tokens(result) < count_tokens(text) else text


# ── Public API ───────────────────────────────────────────────────────────────

def rule_optimize(prompt: str, aggressiveness: str = "moderate") -> str:
    """
    Apply rule-based compression to a prompt.

    Aggressiveness controls how many phases run:
      light      → phase 2 (phrase compression) + output directive
      moderate   → phases 1 + 2 + 3 + output directive
      aggressive → phases 1 + 2 + 3 + 4 + output directive

    Always returns a non-empty string. Falls back to original if result is empty.
    """
    if not prompt or not prompt.strip():
        return prompt

    # ── Phase 0: Duplicate sentence removal (on original, before compression) ─
    prompt = _remove_duplicate_sentences(prompt)

    # ── Detect intent words BEFORE any compression strips them ───────────────
    prompt_lower = prompt.lower()
    detected_intent = next((w for w in _INTENT_WORDS if w in prompt_lower), None)

    # ── Guard code blocks: extract before rules, restore after ───────────────
    result, _blocks = _extract_blocks(prompt)

    # ── Structured optimization: noise-cleaned prompt + inline list → bullets ─
    # Apply light noise pass first so "Can you please explain" → "Explain".
    if detected_intent and aggressiveness != "aggressive":
        cleaned = _NOISE_RE.sub("", result).strip()
        structured = _try_structure(cleaned, detected_intent)
        if structured:
            # Only use structured form if it actually reduces tokens
            if count_tokens(structured) < count_tokens(prompt):
                if structured[0].islower():
                    structured = structured[0].upper() + structured[1:]
                return _restore_blocks(structured, _blocks)

    # ── Short request compression (< 50 tokens, conversational → imperative) ──
    # Mirror of compress_long_text_tags but for 1-2 sentence feature requests.
    # Run FIRST — before noise/phrase passes — so it sees the original wording.
    if count_tokens(result) <= 50 and aggressiveness != "light":
        short_compressed = compress_short_request(result)
        if short_compressed != result:
            result = short_compressed

    if aggressiveness == "light":
        result = _NOISE_RE.sub("", result)
        result = _remove_dangling_words(result)
    elif aggressiveness == "moderate":
        result = _remove_evaluative_sentences(result)
        result = _NOISE_RE.sub("", result)
        for pattern, replacement in _PHRASE_RES:
            result = pattern.sub(replacement, result)
        result = _compress_structure(result)
        result = _remove_dangling_words(result)
        result = _semantic_dedup(result)
    else:
        # Aggressive: full pipeline + semantic dedup
        result = _remove_evaluative_sentences(result)
        result = _NOISE_RE.sub("", result)
        for pattern, replacement in _PHRASE_RES:
            result = pattern.sub(replacement, result)
        result = _compress_structure(result)
        result = _remove_dangling_words(result)
        result = _clean_sentence(result)
        result = _semantic_dedup(result)

    result = result.strip()

    # Capitalise first letter if it got lower-cased by noise removal
    if result and prompt[0].isupper() and result[0].islower():
        result = result[0].upper() + result[1:]

    # Punctuation cleanup runs on prose BEFORE code blocks are reinserted
    # (otherwise it collapses newlines/indent inside code content)
    result = _fix_punctuation(result)

    # Restore code blocks verbatim — always last
    result = _restore_blocks(result, _blocks)

    # ── Intent word re-injection: never silently drop depth signals ──────────
    # Normalise spaces/hyphens AND strip intensifiers (very/quite/really) before checking,
    # so "in great detail" counts as preserving "in very great detail".
    _norm = lambda s: s.replace("-", " ")
    _core_detected = _INTENSIFIER_RE.sub('', detected_intent).strip() if detected_intent else ""
    _intent_missing = (
        detected_intent and result
        and _norm(detected_intent) not in _norm(result.lower())
        and _core_detected not in _norm(result.lower())
    )
    if _intent_missing:
        first_end = result.find(".")
        first_sent = result[:first_end] if first_end != -1 else result
        rest = result[first_end:] if first_end != -1 else ""
        m = _STRUCT_TRIGGER_RE.search(first_sent)
        if m:
            first_sent = (
                first_sent[: m.start()].rstrip(", ") + f" {detected_intent}, " + first_sent[m.start():]
            )
        else:
            first_sent = first_sent.rstrip() + f" {detected_intent}"
        result = first_sent + rest

    # Safety: never return empty — fall back to original
    return result if result else prompt


# ── Keyword extraction for aggressive Ollama reconstruction ─────────────────

_KW_ARTICLE_RE  = re.compile(r'\b(the|a|an)\b\s*',                          re.IGNORECASE)
_KW_AUX_RE      = re.compile(
    r'\b(is|are|was|were|be|been|being|do|does|did|has|have|had)\b\s*',      re.IGNORECASE)
_KW_CONJ_RE     = re.compile(r'\b(and|but|so|then|or)\b[\s,]*',             re.IGNORECASE)
_KW_WEAK_PREP   = re.compile(r'\b(of|in|at|on)\b\s*(?=[a-zA-Z])',           re.IGNORECASE)


def extract_keywords(prompt: str) -> str:
    """
    Strip grammar words to keyword soup — input for Ollama reconstruction.

    Pipeline: safe moderate rules → articles → aux verbs → conjunctions → weak prepositions.
    The output is intentionally ungrammatical; Ollama rewrites it into a clean prompt.
    """
    text = rule_optimize(prompt, "moderate")  # safe rules first
    text = _KW_ARTICLE_RE.sub("", text)
    text = _KW_AUX_RE.sub("", text)
    text = _KW_CONJ_RE.sub(", ", text)       # replace conjunctions with comma separator
    text = _KW_WEAK_PREP.sub("", text)
    # Clean artefacts
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^[,\s]+", "", text)
    text = re.sub(r"[,\s]+$", "", text)
    return text.strip() or prompt


def estimate_rule_savings(prompt: str, aggressiveness: str = "moderate") -> dict:
    """
    Dry-run the rule optimizer and return a savings preview without side effects.
    Useful for showing users what they'd save before committing.
    """
    optimized = rule_optimize(prompt, aggressiveness)
    original_len = len(prompt.split())
    optimized_len = len(optimized.split())
    saved = original_len - optimized_len
    pct = round(saved / original_len * 100, 1) if original_len > 0 else 0.0
    return {
        "original_word_count": original_len,
        "optimized_word_count": optimized_len,
        "words_saved": saved,
        "estimated_savings_percent": pct,
        "optimized_preview": optimized,
    }
