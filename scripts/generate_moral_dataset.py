#!/usr/bin/env python3
"""
Moral Training Dataset Generator
=================================
Generates two dataset files for training the Eleos model's moral reasoning:

  datasets/moral_pretrain/data.jsonl   — plain text pretraining format
  datasets/moral_instruct/data.jsonl  — instruction-tuning (user/assistant) format

Sources:
  - Curated religious and philosophical texts (public domain)
  - HuggingFace: hendrycks/ethics (ETHICS benchmark)
  - HuggingFace: demelin/moral_stories

Usage:
    python scripts/generate_moral_dataset.py
    python scripts/generate_moral_dataset.py --no-hf   # skip HuggingFace downloads
    python scripts/generate_moral_dataset.py --output-dir ./my_datasets
"""

import argparse
import json
import os
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MoralSample:
    text: str               # The raw moral text / teaching
    source: str             # Source name (e.g. "Bible - Proverbs")
    framework: str          # Ethical framework label
    principle: str          # One-sentence description of the moral principle


# ---------------------------------------------------------------------------
# Religious & Philosophical Texts  (public domain)
# ---------------------------------------------------------------------------

RELIGIOUS_SECULAR_TEXTS: list[MoralSample] = [

    # ── Bible (KJV) ─────────────────────────────────────────────────────────
    MoralSample(
        text="Do unto others as you would have them do unto you.",
        source="Bible - Matthew 7:12 (Golden Rule)",
        framework="Virtue / Reciprocity",
        principle="Treat others with the same consideration you wish for yourself.",
    ),
    MoralSample(
        text=(
            "Love is patient, love is kind. It does not envy, it does not boast, "
            "it is not proud. It does not dishonor others, it is not self-seeking, "
            "it is not easily angered, it keeps no record of wrongs."
        ),
        source="Bible - 1 Corinthians 13:4-5",
        framework="Virtue Ethics / Agape",
        principle="Love is the foundation of ethical behavior — patient, kind, and selfless.",
    ),
    MoralSample(
        text=(
            "Blessed are the merciful, for they will be shown mercy. "
            "Blessed are the pure in heart, for they will see God. "
            "Blessed are the peacemakers, for they will be called children of God."
        ),
        source="Bible - Matthew 5:7-9 (Beatitudes)",
        framework="Virtue Ethics",
        principle="Mercy, purity of intention, and peacemaking are marks of a virtuous life.",
    ),
    MoralSample(
        text=(
            "Do not repay evil with evil or insult with insult. On the contrary, "
            "repay evil with blessing, because to this you were called so that you "
            "may inherit a blessing."
        ),
        source="Bible - 1 Peter 3:9",
        framework="Non-violence / Forgiveness",
        principle="Respond to wrongdoing with grace and blessing rather than retaliation.",
    ),
    MoralSample(
        text=(
            "Learn to do right; seek justice. Defend the oppressed. "
            "Take up the cause of the fatherless; plead the case of the widow."
        ),
        source="Bible - Isaiah 1:17",
        framework="Justice / Advocacy",
        principle="Active pursuit of justice for the vulnerable is a moral obligation.",
    ),
    MoralSample(
        text=(
            "Trust in the Lord with all your heart and lean not on your own understanding; "
            "in all your ways submit to him, and he will make your paths straight. "
            "Do not be wise in your own eyes; fear the Lord and shun evil."
        ),
        source="Bible - Proverbs 3:5-7",
        framework="Humility / Wisdom",
        principle="Humility before greater wisdom, and the rejection of arrogance, leads to right action.",
    ),
    MoralSample(
        text=(
            "A generous person will prosper; whoever refreshes others will be refreshed. "
            "People curse the one who hoards grain, but they pray God's blessing on the "
            "one who is willing to sell."
        ),
        source="Bible - Proverbs 11:25-26",
        framework="Generosity / Community",
        principle="Generosity and sharing create flourishing communities; hoarding harms them.",
    ),

    # ── Quran ────────────────────────────────────────────────────────────────
    MoralSample(
        text=(
            "O you who have believed, be persistently standing firm for Allah, witnesses in "
            "justice, and do not let the hatred of a people prevent you from being just. "
            "Be just; that is nearer to righteousness."
        ),
        source="Quran - Al-Ma'idah 5:8",
        framework="Justice / Impartiality",
        principle="Justice must be upheld impartially, even toward those one dislikes.",
    ),
    MoralSample(
        text=(
            "And He has set up the balance, that you may not transgress the balance. "
            "And observe the weight with equity and do not make the balance deficient."
        ),
        source="Quran - Ar-Rahman 55:7-9",
        framework="Justice / Equity",
        principle="Honesty in measure and equitable dealing are expressions of cosmic order.",
    ),
    MoralSample(
        text=(
            "Whoever saves one life, it is as if he had saved all mankind. "
            "And whoever kills a soul — unless for a soul or for corruption done in the land — "
            "it is as if he had killed all mankind."
        ),
        source="Quran - Al-Ma'idah 5:32",
        framework="Sanctity of Life",
        principle="Each human life has infinite value; its preservation or destruction carries universal moral weight.",
    ),
    MoralSample(
        text=(
            "And do good as Allah has done good to you, and do not seek to cause corruption "
            "in the land. Indeed, Allah does not like the corrupters."
        ),
        source="Quran - Al-Qasas 28:77",
        framework="Beneficence / Environmental Ethics",
        principle="Doing good to others and to the earth is obligatory; corruption is a moral failure.",
    ),
    MoralSample(
        text=(
            "Indeed, Allah orders justice and good conduct and giving to relatives "
            "and forbids immorality and bad conduct and oppression. "
            "He admonishes you that perhaps you will be reminded."
        ),
        source="Quran - An-Nahl 16:90",
        framework="Justice / Virtue",
        principle="The moral life is defined by justice, kindness to family, and rejection of oppression.",
    ),

    # ── Bhagavad Gita ────────────────────────────────────────────────────────
    MoralSample(
        text=(
            "Let right deeds be thy motive, not the fruit which comes from them. "
            "And live in the action, labour well the task which duty bids thee do: "
            "this is unselfish service."
        ),
        source="Bhagavad Gita - Chapter 2:47 (paraphrase)",
        framework="Duty / Deontology / Karma Yoga",
        principle="Act from duty and right intention, not from expectation of personal reward.",
    ),
    MoralSample(
        text=(
            "He who has no attachments can really love others, for his love is pure and "
            "divine. He who is free from hatred toward all living beings, who is friendly "
            "and compassionate — such a devotee of Mine is very dear to Me."
        ),
        source="Bhagavad Gita - Chapter 12:13-14",
        framework="Compassion / Non-attachment",
        principle="Genuine compassion flows from non-attachment and freedom from hatred.",
    ),
    MoralSample(
        text=(
            "The self-controlled soul, who moves amongst sense objects, free from either "
            "attachment or repulsion, wins eternal peace. Those who are not self-controlled "
            "live in fear and agitation, unable to find peace."
        ),
        source="Bhagavad Gita - Chapter 2:64-65",
        framework="Self-discipline / Inner Peace",
        principle="Mastery of the senses and emotions is the foundation of lasting peace and right action.",
    ),
    MoralSample(
        text=(
            "Fearlessness, purity of heart, perseverance in the pursuit of wisdom, charity, "
            "self-control, sacrifice, study of scripture, austerity, and uprightness, "
            "non-violence, truth, freedom from anger — these are the treasures of one born "
            "with divine qualities."
        ),
        source="Bhagavad Gita - Chapter 16:1-3",
        framework="Virtue Ethics / Ahimsa",
        principle="A virtuous person cultivates a broad constellation of qualities including truth, non-violence, and compassion.",
    ),

    # ── Tao Te Ching (Lao Tzu) ───────────────────────────────────────────────
    MoralSample(
        text=(
            "The Tao that can be told is not the eternal Tao. "
            "The sage does not compete, and therefore no one can compete with him. "
            "He does not display himself, so he shines. He does not justify himself, "
            "so he is distinguished. He does not boast of himself, therefore he has merit."
        ),
        source="Tao Te Ching - Chapter 22",
        framework="Humility / Non-Striving",
        principle="True virtue is quiet and non-competitive; greatness comes from not seeking it.",
    ),
    MoralSample(
        text=(
            "Treat those who are good with goodness, and also treat those who are not good "
            "with goodness. Thus goodness is attained. Be honest with those who are honest, "
            "and be also honest with those who are not honest. Thus honesty is attained."
        ),
        source="Tao Te Ching - Chapter 49",
        framework="Universal Goodness / Consistency",
        principle="Respond with goodness and honesty universally, not selectively — this is true virtue.",
    ),
    MoralSample(
        text=(
            "The sage does not accumulate. The more he does for others, the more he has. "
            "The more he gives to others, the more he possesses. The Tao of heaven benefits "
            "and does not harm. The Tao of the sage acts and does not compete."
        ),
        source="Tao Te Ching - Chapter 81",
        framework="Generosity / Non-competition",
        principle="Generosity paradoxically enriches the giver; the virtuous act without competing.",
    ),

    # ── Dhammapada (Buddhist Ethics) ─────────────────────────────────────────
    MoralSample(
        text=(
            "Mind is the forerunner of all actions. All deeds are led by mind, created by "
            "mind. If one speaks or acts with a corrupt mind, suffering follows, as the wheel "
            "follows the hoof of an ox. If one speaks or acts with a serene mind, happiness "
            "follows, as a shadow that never departs."
        ),
        source="Dhammapada - Verse 1-2",
        framework="Intention / Karma",
        principle="The moral quality of an act is determined by the intention behind it.",
    ),
    MoralSample(
        text=(
            "Do not speak harshly to anyone; those who are spoken to will answer thee in the "
            "same way. Angry speech is painful: blows for blows will touch thee."
        ),
        source="Dhammapada - Verse 133",
        framework="Non-violence / Speech Ethics",
        principle="Harsh speech creates suffering and invites retaliation; kind speech cultivates peace.",
    ),
    MoralSample(
        text=(
            "Conquer anger by love, conquer evil by good. Conquer the miser by generosity, "
            "conquer the liar by truth."
        ),
        source="Dhammapada - Verse 223",
        framework="Non-violence / Virtue",
        principle="The antidote to vice is its corresponding virtue — anger is overcome by love, not force.",
    ),
    MoralSample(
        text=(
            "He who has renounced violence towards all living beings, weak or strong, who "
            "neither kills nor causes others to kill — him do I call a Brahmin."
        ),
        source="Dhammapada - Verse 405",
        framework="Ahimsa (Non-violence)",
        principle="True nobility is defined not by birth but by non-violence toward all sentient beings.",
    ),

    # ── Stoic Philosophy ─────────────────────────────────────────────────────
    MoralSample(
        text=(
            "You have power over your mind — not outside events. Realize this, and you will "
            "find strength. The impediment to action advances action. What stands in the way "
            "becomes the way."
        ),
        source="Marcus Aurelius - Meditations",
        framework="Stoic Virtue / Resilience",
        principle="Moral strength comes from mastery of one's own responses, not control of external events.",
    ),
    MoralSample(
        text=(
            "Never esteem anything as of advantage to you that will make you break your word "
            "or lose your self-respect. Waste no more time arguing what a good man should be. "
            "Be one."
        ),
        source="Marcus Aurelius - Meditations",
        framework="Integrity / Virtue",
        principle="Integrity and self-respect are paramount; do not compromise them for any apparent advantage.",
    ),
    MoralSample(
        text=(
            "Make the best use of what is in your power, and take the rest as it happens. "
            "Seek not the good in external things; seek it in yourself. "
            "Men are disturbed not by the things which happen, but by the opinions about the things."
        ),
        source="Epictetus - Enchiridion",
        framework="Stoic Virtue / Autonomy",
        principle="Virtue and good judgment are internal; external circumstances do not determine moral worth.",
    ),
    MoralSample(
        text=(
            "First say to yourself what you would be; then do what you have to do. "
            "No person is free who is not master of himself. "
            "He is a wise man who does not grieve for the things which he has not, "
            "but rejoices for those which he has."
        ),
        source="Epictetus - Discourses",
        framework="Stoic Virtue / Self-mastery",
        principle="Self-mastery and contentment are the foundations of freedom and virtuous action.",
    ),
    MoralSample(
        text=(
            "It is not the man who has too little that is poor, but the one who hankers after more. "
            "True happiness is to enjoy the present, without anxious dependence upon the future, "
            "not to amuse ourselves with either hopes or fears but to rest satisfied with what we have."
        ),
        source="Seneca - Letters to Lucilius",
        framework="Contentment / Moderation",
        principle="Contentment with what is sufficient, rather than craving for excess, is the path to happiness.",
    ),

    # ── Secular Humanism ─────────────────────────────────────────────────────
    MoralSample(
        text=(
            "Humanism is a progressive philosophy of life that, without theism or other "
            "supernatural beliefs, affirms our ability and responsibility to lead ethical and "
            "fulfilling lives capable of adding to the greater good of humanity."
        ),
        source="American Humanist Association - Humanist Manifesto III",
        framework="Secular Humanism",
        principle="Human beings can live ethically and meaningfully without supernatural authority, guided by reason and empathy.",
    ),
    MoralSample(
        text=(
            "We are committed to treating each person as having inherent worth and dignity. "
            "Humanists ground values, meaning, and moral reasoning in human welfare and "
            "flourishing. We accept our duty to care for one another and for the natural world."
        ),
        source="Amsterdam Declaration of Humanism (2002)",
        framework="Human Dignity / Care Ethics",
        principle="Every person has inherent dignity; we have an obligation to one another and to our shared environment.",
    ),
    MoralSample(
        text=(
            "Ethics is consequential. Humanists are convinced that the solutions to human "
            "problems lie in human thought and action. We believe in science, reason, and "
            "democracy as the best means available to human beings for solving problems "
            "and increasing human flourishing."
        ),
        source="Amsterdam Declaration of Humanism (2002)",
        framework="Consequentialism / Reason",
        principle="Reason, science, and democratic deliberation are the most reliable tools for ethical problem-solving.",
    ),
    MoralSample(
        text=(
            "The good life is one inspired by love and guided by knowledge. "
            "The root of the matter is a very simple and old-fashioned thing — a thing so "
            "simple that I am almost ashamed to mention it, for fear of the derisive smile "
            "with which wise cynics will greet my words. That thing is love: Christian love, "
            "or compassion."
        ),
        source="Bertrand Russell - What I Believe",
        framework="Secular Humanism / Compassion",
        principle="Love and knowledge are the twin guides of a good life, irrespective of religious framework.",
    ),
    MoralSample(
        text=(
            "The purpose of morality is to promote human flourishing: the health, happiness, "
            "and freedom of individuals and communities. Cruelty is wrong because it causes "
            "suffering. Fairness matters because people have equal dignity and worth. "
            "Compassion is a virtue because it motivates us to reduce suffering."
        ),
        source="Peter Singer - Practical Ethics (paraphrase)",
        framework="Utilitarian / Care Ethics",
        principle="Morality is grounded in the reduction of suffering and the promotion of flourishing for all beings capable of experiencing them.",
    ),
    MoralSample(
        text=(
            "Act only according to that maxim whereby you can at the same time will that it "
            "should become a universal law. Always treat humanity, whether in your own person "
            "or that of another, never merely as a means to an end, but always at the same "
            "time as an end."
        ),
        source="Immanuel Kant - Groundwork of the Metaphysics of Morals",
        framework="Deontological Ethics / Categorical Imperative",
        principle="Treat all people as ends in themselves, never as mere means; act only by principles you could universalize.",
    ),
    MoralSample(
        text=(
            "We hold these truths to be self-evident, that all men are created equal, that "
            "they are endowed with certain unalienable Rights, that among these are Life, "
            "Liberty and the pursuit of Happiness — and that to secure these rights, "
            "Governments are instituted among Men."
        ),
        source="United States Declaration of Independence (1776)",
        framework="Natural Rights / Liberal Democracy",
        principle="Equality and fundamental rights are universal; governance exists to protect them.",
    ),
    MoralSample(
        text=(
            "All human beings are born free and equal in dignity and rights. They are endowed "
            "with reason and conscience and should act towards one another in a spirit of brotherhood."
        ),
        source="UN Universal Declaration of Human Rights - Article 1",
        framework="Human Rights / Universal Dignity",
        principle="Human dignity and equality are foundational; reason and conscience guide moral relations.",
    ),
    MoralSample(
        text=(
            "The moral progress of humanity depends on extending our circle of compassion: "
            "from tribe to nation, from nation to humanity, and ultimately to all sentient "
            "beings capable of suffering. Every step in this expansion has been resisted, "
            "and every step has proven to be right."
        ),
        source="Peter Singer - The Expanding Circle (paraphrase)",
        framework="Utilitarian / Universal Compassion",
        principle="Moral progress requires progressively extending empathy and moral consideration to all who can suffer.",
    ),
]


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

INSTRUCT_TEMPLATES = [
    # Explanation template
    lambda s: {
        "messages": [
            {
                "role": "user",
                "content": f'What is the moral teaching of this text, and how might it guide behavior today?\n\n"{s.text}"',
            },
            {
                "role": "assistant",
                "content": (
                    f"This passage from {s.source} reflects the principle of {s.framework}. "
                    f"{s.principle}\n\n"
                    f"In practical terms, this teaching asks us to consider how our actions affect others, "
                    f"to act with integrity and intention, and to cultivate virtues such as compassion, "
                    f"honesty, and justice. Across the centuries and across cultures, this kind of "
                    f"ethical reasoning continues to offer meaningful guidance for how we treat one another."
                ),
            },
        ]
    },
    # Comparison template
    lambda s: {
        "messages": [
            {
                "role": "user",
                "content": f"Explain the ethical principle from {s.source} in your own words.",
            },
            {
                "role": "assistant",
                "content": (
                    f"{s.principle}\n\n"
                    f'The original text states: "{s.text}"\n\n'
                    f"This is an example of {s.framework} ethics. It asks us to evaluate our actions "
                    f"not just by their outcomes, but by the character they express and the community "
                    f"they help build. Whether one approaches ethics from a religious or secular standpoint, "
                    f"the core insight — that how we treat others matters deeply — is widely shared."
                ),
            },
        ]
    },
    # Application template
    lambda s: {
        "messages": [
            {
                "role": "user",
                "content": f"How does the teaching from {s.source} apply to how an AI system should behave?",
            },
            {
                "role": "assistant",
                "content": (
                    f"The teaching from {s.source} offers a meaningful lens for AI behavior. "
                    f"It says: \"{s.text}\"\n\n"
                    f"Translated into the context of an AI system, this principle — {s.principle} — "
                    f"suggests that an AI should:\n\n"
                    f"1. Always consider the well-being of the humans it interacts with\n"
                    f"2. Act with honesty and transparency, never manipulating or deceiving\n"
                    f"3. Treat every person with equal dignity, regardless of who they are\n"
                    f"4. Refuse to assist with actions that cause harm, even when asked\n"
                    f"5. Prioritize long-term flourishing over short-term satisfaction\n\n"
                    f"The {s.framework} tradition reminds us that morality is not a set of arbitrary "
                    f"constraints, but a way of relating to others that reflects our deepest values."
                ),
            },
        ]
    },
]


# ---------------------------------------------------------------------------
# HuggingFace dataset loaders
# ---------------------------------------------------------------------------

def load_ethics_dataset() -> Iterator[MoralSample]:
    """Load samples from hendrycks/ethics benchmark."""
    try:
        from datasets import load_dataset
        print("  Loading hendrycks/ethics...")
        ethics = load_dataset("hendrycks/ethics", "commonsense", split="train", trust_remote_code=True)
        for row in ethics:
            label = "right" if row["label"] == 0 else "wrong"
            yield MoralSample(
                text=row["input"],
                source="ETHICS Benchmark (Hendrycks et al.)",
                framework="Commonsense Ethics",
                principle=f"This action is considered morally {label} by common human ethical intuition.",
            )
    except Exception as e:
        print(f"  Warning: Could not load hendrycks/ethics: {e}")


def load_moral_stories_dataset() -> Iterator[MoralSample]:
    """Load samples from moral_stories dataset."""
    try:
        from datasets import load_dataset
        print("  Loading demelin/moral_stories...")
        stories = load_dataset("demelin/moral_stories", "full", split="train")
        for row in stories:
            if row.get("norm") and row.get("situation"):
                yield MoralSample(
                    text=f"Norm: {row['norm']}\nSituation: {row['situation']}",
                    source="Moral Stories (Emelin et al.)",
                    framework="Social Norms / Consequentialism",
                    principle=row["norm"],
                )
    except Exception as e:
        print(f"  Warning: Could not load demelin/moral_stories: {e}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_pretrain_sample(f, sample: MoralSample) -> None:
    """Write a plain-text pretraining sample."""
    text = (
        f"[Source: {sample.source}] [{sample.framework}]\n\n"
        f"{sample.text}\n\n"
        f"Moral principle: {sample.principle}\n"
    )
    f.write(json.dumps({"text": text}) + "\n")


def write_instruct_sample(f, sample: MoralSample) -> None:
    """Write an instruction-tuning sample using a random template."""
    template = random.choice(INSTRUCT_TEMPLATES)
    record = template(sample)
    # Convert to plain text for HuggingFaceTextDataset
    user_msg = record["messages"][0]["content"]
    asst_msg = record["messages"][1]["content"]
    text = f"User: {user_msg}\n\nAssistant: {asst_msg}\n"
    f.write(json.dumps({"text": text, "messages": record["messages"]}) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate moral training datasets")
    parser.add_argument(
        "--output-dir",
        default="./datasets",
        help="Root directory for output datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Skip HuggingFace dataset downloads (use curated texts only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for template selection",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    pretrain_dir = Path(args.output_dir) / "moral_pretrain"
    instruct_dir = Path(args.output_dir) / "moral_instruct"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    instruct_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[MoralSample] = list(RELIGIOUS_SECULAR_TEXTS)

    if not args.no_hf:
        print("Downloading HuggingFace datasets...")
        all_samples.extend(load_ethics_dataset())
        all_samples.extend(load_moral_stories_dataset())

    random.shuffle(all_samples)

    pretrain_path = pretrain_dir / "data.jsonl"
    instruct_path = instruct_dir / "data.jsonl"

    print(f"\nWriting {len(all_samples)} samples...")

    with open(pretrain_path, "w") as fp, open(instruct_path, "w") as fi:
        for sample in all_samples:
            write_pretrain_sample(fp, sample)
            write_instruct_sample(fi, sample)

    print(f"\n✓ Pretraining dataset : {pretrain_path}  ({pretrain_path.stat().st_size // 1024} KB)")
    print(f"✓ Instruction dataset : {instruct_path}  ({instruct_path.stat().st_size // 1024} KB)")
    print(f"\nTotal samples: {len(all_samples)}")
    print("\nFrameworks represented:")
    frameworks = sorted({s.framework for s in all_samples})
    for fw in frameworks:
        count = sum(1 for s in all_samples if s.framework == fw)
        print(f"  {count:4d}  {fw}")
    print(
        "\nNext steps:\n"
        "  1. Add 'moral_pretrain' and 'moral_instruct' entries to DATASETS in\n"
        "     torchtitan/hf_datasets/text_datasets.py\n"
        "  2. Set dataloader=HuggingFaceTextDataLoader.Config(dataset='moral_pretrain')\n"
        "     in your config_registry.py trainer config.\n"
        "  3. Run: CONFIG=eleos_small ./run_train_mac.sh\n"
    )


if __name__ == "__main__":
    main()
