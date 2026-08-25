"""A tiny deterministic development corpus.

Public-domain sources (all published well before 1900, US public domain) plus a
handful of synthetic sentences written for this repository. Long enough to build
prefix+canvas pairs for the overfit test, small enough to live in the repo.

The synthetic block exists because the overfit-one-batch test needs examples with
predictable local structure, so that a failure to memorise is unambiguous.
"""

from __future__ import annotations

#: Public domain: Jane Austen, *Pride and Prejudice* (1813); Lewis Carroll,
#: *Alice's Adventures in Wonderland* (1865); Herman Melville, *Moby-Dick* (1851).
PUBLIC_DOMAIN: list[str] = [
    "It is a truth universally acknowledged, that a single man in possession of a "
    "good fortune, must be in want of a wife. However little known the feelings or "
    "views of such a man may be on his first entering a neighbourhood, this truth is "
    "so well fixed in the minds of the surrounding families, that he is considered as "
    "the rightful property of some one or other of their daughters.",
    "Alice was beginning to get very tired of sitting by her sister on the bank, and "
    "of having nothing to do: once or twice she had peeped into the book her sister "
    "was reading, but it had no pictures or conversations in it, and what is the use "
    "of a book, thought Alice, without pictures or conversations?",
    "Call me Ishmael. Some years ago, never mind how long precisely, having little or "
    "no money in my purse, and nothing particular to interest me on shore, I thought I "
    "would sail about a little and see the watery part of the world. It is a way I have "
    "of driving off the spleen and regulating the circulation.",
    "There was nothing so very remarkable in that; nor did Alice think it so very much "
    "out of the way to hear the Rabbit say to itself, Oh dear! Oh dear! I shall be late! "
    "But when the Rabbit actually took a watch out of its waistcoat pocket, and looked at "
    "it, and then hurried on, Alice started to her feet.",
]

#: Synthetic, written for this repo. Short, highly regular, easy to memorise.
SYNTHETIC: list[str] = [
    "The cat sat on the windowsill watching the rain fall steadily on the quiet street "
    "below, and the street below stayed quiet while the rain kept falling on the cat's "
    "window and on the roofs of every house along the row.",
    "The engineer measured the beam, wrote the number in her notebook, measured the beam "
    "again, and wrote the second number underneath the first, because a single measurement "
    "of a single beam is never a measurement of anything at all.",
    "Monday came before Tuesday, Tuesday came before Wednesday, Wednesday came before "
    "Thursday, and Thursday came before Friday, and in this way the whole week arranged "
    "itself in the only order the week has ever agreed to take.",
    "A model that predicts the next token is not the same as a model that repairs a "
    "damaged token, and the difference between predicting and repairing is exactly the "
    "difference this experiment is trying to measure.",
]

DEV_TEXTS: list[str] = PUBLIC_DOMAIN + SYNTHETIC
