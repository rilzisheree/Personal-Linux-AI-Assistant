"""Deterministic, leakage-resistant data generation for the Lura router.

The generator keeps concepts grouped together when splitting. A concept is
therefore either in train, validation, or test, which prevents near-duplicate
phrases from making the unseen-test score look better than it is.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


LABELS = ("SIMPLE", "FUNCTION", "REASONING")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    phrases: tuple[str, ...]
    voice_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouterExample:
    id: str
    text: str
    label: str
    scenario: str
    split: str
    source: str = "generated"


_SIMPLE = (
    Scenario("greeting", "SIMPLE", (
        "Hello", "Hi", "Hey there", "Good morning", "Good evening", "Hi Luna",
        "Hello Luna", "Hey Luna", "Are you there?", "Are you awake?",
        "Just saying hello", "Just checking in", "Hello again", "Hey there Luna",
        "Hi there", "Morning Luna", "Evening Luna", "Hey, are you around?",
        "Luna, can you hear me?", "I wanted to say hello", "A quick hello",
        "Nice to see you", "How's your day going?", "Hello for now",
    )),
    Scenario("thanks", "SIMPLE", (
        "Thanks", "Thank you", "I appreciate it", "That helps, thank you",
        "Thanks a lot", "I really appreciate that", "Much appreciated",
        "That's helpful", "You're helpful, thanks", "Thank you Luna",
        "Thanks for that", "I appreciate your help", "Perfect, thank you",
        "That's all, thanks", "You've been helpful", "Cheers, Luna",
        "Thanks again", "I owe you one", "Great, thanks", "Helpful as always",
        "That answers it, thanks", "Thanks for listening", "All good, thank you",
        "I appreciate the answer",
    )),
    Scenario("goodbye", "SIMPLE", (
        "Goodbye", "Bye", "See you later", "I'm done for now", "Talk to you later",
        "That's all for now", "I have to go", "Good night", "See you soon",
        "Catch you later", "I'm finished", "You can go quiet now", "Bye Luna",
        "Goodbye for now", "I'll be back later", "End this chat", "I'm signing off",
        "Have a good night", "Take care", "Until next time", "No more for now",
        "I'm done chatting", "That's enough, thanks", "See you around",
    )),
    Scenario("wellbeing", "SIMPLE", (
        "How are you?", "Are you doing okay?", "How's it going?", "How are things?",
        "How do you feel?", "Are you well?", "Everything okay?", "How is your day?",
        "Are you having a good day?", "How have you been?", "You doing alright?",
        "How are you today?", "Are you okay Luna?", "How's everything going?",
        "Is everything good?", "How are things with you?", "Are you feeling fine?",
        "How is life?", "You alright there?", "How's your morning?",
        "How's your evening?", "Are you in a good mood?", "How have things been?",
        "Luna, how are you?",
    )),
    Scenario("arithmetic", "SIMPLE", ("What is 17 plus 8?", "What's 6 times 9?", "How much is 144 divided by 12?")),
    Scenario("general_fact", "SIMPLE", ("What is the capital of France?", "How many days are in a week?", "What color is made by mixing blue and yellow?")),
    Scenario("definition", "SIMPLE", ("Give me a one-sentence definition of gravity.", "What does the word ephemeral mean?", "Define photosynthesis briefly.")),
    Scenario("short_joke", "SIMPLE", ("Tell me a short joke.", "Make me laugh with one quick joke.", "Give me a tiny joke, nothing long.")),
    Scenario("affirmation", "SIMPLE", (
        "Confirm that you are listening.", "Say hello back.", "Just acknowledge me.",
        "Let me know you heard me.", "Can you confirm you're there?",
        "Give me a quick acknowledgement.", "Tell me you're listening.",
        "Respond so I know the microphone works.", "Just say something.",
        "Acknowledge this message.", "Can you hear what I'm saying?",
        "Let me know you're awake.", "Say anything at all.", "Give me a sign.",
        "Check in with me.", "Respond with a quick hello.", "Are you listening?",
        "Tell me you can hear me.", "Just confirm receipt.", "Answer with yes.",
        "Can you respond briefly?", "Let me know you're available.",
        "Please acknowledge me.", "Are you still there?",
    )),
    Scenario("spelling", "SIMPLE", ("How do you spell necessary?", "Spell the word restaurant.", "What's the plural of mouse?")),
    Scenario("unit_fact", "SIMPLE", ("How many minutes are in an hour?", "How many centimeters are in a meter?", "How many sides does a hexagon have?")),
    Scenario("small_conversion", "SIMPLE", ("Convert five dollars to cents.", "How many seconds are in two minutes?", "What is 3 kilometers in meters?")),
)

_FUNCTION = (
    Scenario("open_app", "FUNCTION", ("Open Discord", "Launch my file manager", "Start Firefox", "Can you bring up Spotify?"), ("discord", "firefox", "spotify")),
    Scenario("close_app", "FUNCTION", ("Close Spotify", "Quit Firefox", "Please shut Discord", "Stop the browser"), ("spotify", "firefox", "discord")),
    Scenario("restart_app", "FUNCTION", ("Restart Firefox", "Reopen Discord from scratch", "Restart my browser", "Can you relaunch Spotify?"), ("firefox", "discord", "spotify")),
    Scenario("screenshot", "FUNCTION", ("Take a screenshot", "Capture my screen", "Show me what's on my desktop", "Grab a screenshot now")),
    Scenario("cpu", "FUNCTION", ("What is my CPU usage?", "Check the current processor load", "How busy is my CPU right now?", "Is my processor under heavy load?"), ("cpu",)),
    Scenario("memory", "FUNCTION", ("How much memory am I using?", "Check my RAM usage", "What is using the most memory?", "Tell me the current memory load"), ("ram", "memory")),
    Scenario("volume", "FUNCTION", ("How loud is my computer right now?", "Check the current volume", "Turn up the volume", "Mute my computer"), ("volume",)),
    Scenario("windows", "FUNCTION", ("List my open windows", "What windows do I have open?", "Show the applications on my desktop", "Which windows are currently visible?")),
    Scenario("website", "FUNCTION", ("Open https://example.com", "Go to the example website", "Open this URL in my browser: https://example.com", "Navigate to example.com"), ("website",)),
    Scenario("file_search", "FUNCTION", ("Find the error log on my computer", "Search my files for config.json", "Look for the latest PDF in my home folder", "Read the notes file from my desktop"), ("file",)),
    Scenario("system_status", "FUNCTION", ("What is my computer status?", "Check my battery and system health", "Show live system information", "Is my machine okay right now?")),
    Scenario("power", "FUNCTION", ("Shut down my PC", "Restart the computer", "Put the machine to sleep", "Power off this computer"), ("computer",)),
)

_REASONING = (
    Scenario("coding", "REASONING", ("Write a Python program that monitors CPU temperature.", "Build a small bash backup script.", "Help me implement a parser for these records.")),
    Scenario("debugging", "REASONING", ("Debug this authentication error and propose a fix.", "Why does this code throw an exception?", "Find the bug in this algorithm.")),
    Scenario("planning", "REASONING", ("Plan a seven-day trip through Japan.", "Help me organize a dinner party.", "Make a realistic study plan for next month.")),
    Scenario("explanation", "REASONING", ("Explain why black holes evaporate.", "Teach me how photosynthesis works.", "Explain this networking concept in detail.")),
    Scenario("comparison", "REASONING", ("Compare SQLite and PostgreSQL for this project.", "What are the tradeoffs between REST and GraphQL?", "Contrast these two database designs.")),
    Scenario("design", "REASONING", ("Help me design an application for tracking habits.", "Design a database schema for a multi-tenant app.", "How should I organize this software project?")),
    Scenario("analysis", "REASONING", ("Analyze these logs and identify the likely cause.", "Review these numbers and find the trend.", "What conclusions can you draw from this data?")),
    Scenario("writing", "REASONING", ("Improve this paragraph without changing its meaning.", "Draft a professional reply to this message.", "Rewrite this explanation so it is clearer.")),
    Scenario("research", "REASONING", ("Research battery technology and summarize the tradeoffs.", "Find the main arguments for and against remote work.", "Give me a sourced overview of this topic.")),
    Scenario("multi_step", "REASONING", ("Solve this multi-step logic puzzle and explain each step.", "Work through this problem carefully from the beginning.", "Break this complicated task into a sequence of actions.")),
    Scenario("ambiguous", "REASONING", ("Make it better.", "I have a complicated problem; help me figure it out.", "Something is wrong with my project and I don't know where to start.")),
    Scenario("optimization", "REASONING", ("How can I make this code faster?", "Optimize this query and explain the changes.", "Suggest a more reliable approach with reasons.")),
)

SCENARIOS: tuple[Scenario, ...] = _SIMPLE + _FUNCTION + _REASONING

_PREFIXES = ("", "Luna, ", "Hey Luna, ", "Could you ", "Please ", "I need you to ", "Can you ")
_SUFFIXES = ("", " please", " when you can", " for me", " right now")
_VOICE_REPLACEMENTS = {
    "discord": ("discourd", "dis cord"),
    "firefox": ("fire fox", "firefocks"),
    "spotify": ("spot ify", "spotyfy"),
    "cpu": ("c p u", "see pee you"),
    "ram": ("r a m", "ran"),
    "volume": ("volum", "vollume"),
    "screenshot": ("screen shot", "screenshott"),
    "browser": ("browsr", "browse er"),
}


def _split_for_scenario(scenario: Scenario, label_index: int) -> str:
    # Six train, three validation, and three test scenarios per label. Keeping
    # concepts disjoint is stricter than a random row-level split.
    position = label_index % 12
    if position < 6:
        return "train"
    if position < 9:
        return "validation"
    return "test"


def _voice_variant(text: str, rng: random.Random, terms: Iterable[str]) -> str:
    if not terms or rng.random() > 0.34:
        return text
    for term in terms:
        replacements = _VOICE_REPLACEMENTS.get(term, ())
        if replacements and term in text.casefold():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            return pattern.sub(rng.choice(replacements), text, count=1)
    return text


def _make_text(scenario: Scenario, rng: random.Random) -> tuple[str, str]:
    phrase = rng.choice(scenario.phrases)
    if scenario.key in {"greeting", "thanks", "goodbye", "wellbeing", "affirmation"}:
        # Greetings and acknowledgements should not be combined with command
        # scaffolding such as "Could you" or an unrelated "for me" suffix.
        prefixes = ("", "Luna, ", "Hey Luna, ")
        suffixes = ("",)
    elif phrase.endswith("?") or phrase.startswith(("What ", "How ", "Why ", "Are ", "Can ")):
        # Avoid forms such as "Can you what is my CPU usage?". A wake word or
        # a light trailing please remains natural for questions.
        prefixes = (
            "", "Luna, ", "Hey Luna, ", "Quick question: ",
            "I was wondering, ", "Can you tell me ", "Do you know ",
            "Could you tell me ",
        )
        suffixes = ("", " please", " if you can", " real quick")
    else:
        prefixes = _PREFIXES
        suffixes = _SUFFIXES
    prefix = rng.choice(prefixes)
    suffix = rng.choice(suffixes)
    if prefix in {
        "Could you ",
        "Please ",
        "I need you to ",
        "Can you ",
        "Can you tell me ",
        "Do you know ",
        "Could you tell me ",
    }:
        phrase = phrase[:1].lower() + phrase[1:]
    text = f"{prefix}{phrase}{suffix}".strip()
    text = _voice_variant(text, rng, scenario.voice_terms)
    if rng.random() < 0.08:
        text = text.replace("?", "").replace(".", "")
    if rng.random() < 0.05:
        text = text.replace("the ", "da ", 1)
    return text, "voice_variant" if text != f"{prefix}{phrase}{suffix}".strip() else "generated"


def generate_examples(
    *,
    seed: int = 2026,
    train_per_scenario: int = 60,
    validation_per_scenario: int = 50,
    test_per_scenario: int = 70,
) -> list[RouterExample]:
    """Generate balanced examples with scenario-level split isolation."""
    counts = {"train": train_per_scenario, "validation": validation_per_scenario, "test": test_per_scenario}
    examples: list[RouterExample] = []
    label_positions = {label: 0 for label in LABELS}
    for scenario_index, scenario in enumerate(SCENARIOS):
        split = _split_for_scenario(scenario, label_positions[scenario.label])
        label_positions[scenario.label] += 1
        rng = random.Random(seed + scenario_index * 1009)
        seen: set[str] = set()
        target = counts[split]
        attempts = 0
        while len(seen) < target and attempts < target * 30:
            attempts += 1
            text, source = _make_text(scenario, rng)
            folded = re.sub(r"\s+", " ", text.casefold())
            if folded in seen:
                continue
            seen.add(folded)
            examples.append(
                RouterExample(
                    id=f"{split}-{scenario.key}-{len(seen):04d}",
                    text=text,
                    label=scenario.label,
                    scenario=scenario.key,
                    split=split,
                    source=source,
                )
            )
        if len(seen) != target:
            raise RuntimeError(f"Could not generate {target} unique examples for {scenario.key}")
    return examples


def write_dataset(output_dir: Path, examples: Iterable[RouterExample]) -> dict:
    """Write split JSONL files and a manifest, returning aggregate counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    by_split = {split: [] for split in SPLITS}
    for example in examples:
        by_split[example.split].append(asdict(example))
    summary: dict = {"labels": list(LABELS), "splits": {}, "scenario_split": {}}
    for split, rows in by_split.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        label_counts = {label: sum(row["label"] == label for row in rows) for label in LABELS}
        summary["splits"][split] = {"examples": len(rows), "labels": label_counts}
        for row in rows:
            summary["scenario_split"][row["scenario"]] = split
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary