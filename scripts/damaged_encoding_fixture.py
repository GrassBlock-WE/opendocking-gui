"""Build the encoding damage `check_text_encoding.py` exists to reject.

Run:  F:\\python310\\python.exe scripts\\damaged_encoding_fixture.py --out DIR

# What this is for

`check_text_encoding.py` asks whether any file in this tree still carries the bytes
a PowerShell 5.1 round trip destroys. Run on a clean tree it prints a confident
`0 damaged`, and a confident zero from a predicate nobody has ever watched refuse
anything is not evidence -- it is the absence of evidence wearing a verdict. So
this builds the damage, and the gate is required to name it.

The builder is a **deliverable, not a gate**: it defines no check of its own, and
`check_scripts_declare.py` classifies it that way structurally, by finding the
`run_script("damaged_encoding_fixture.py")` call in the gate's own source. A file
like this that nothing launches is a fixture with no oracle saying it was ever
built, and its output is a directory path -- not a number anybody can check by
reading it. `stale_install_fixture.py` is the precedent, and `release_tree_rule.py`
is the other shape of module that is not a gate.

# One fixture per signature, and why that is not padding

The gate holds nine independent predicates. Each can be broken without touching the
other eight, and one fixture that exercised two of them at once would go green
against a gate with either one switched off -- the same lesson
`installed_copy_check.py` records after its package-initialiser fixture proved
only half of a two-part claim. So there is **one damaged file per signature**,
**two negative controls that must be accepted**, and **a deliberately
mis-encoded file that must be accepted**.

Ten damaged files against nine predicates, because the multi-byte class has
two halves with different sources and the one this tree cannot produce needs
its own fixture. A count that is not a count of the rule would be the thing
this builder exists to avoid, so the extra file is declared as a signature in
`DAMAGE.txt` in its own right rather than folded into a shared entry.

**The second negative control is new, and it exists because of what the ninth
predicate hunts.** The cp1252 image of one character is two characters, and
this repository writes `angstrom` followed by a superscript three 52 times
across 10 real files as measured on 2026-10-03 -- the same two characters a
damaged `U+00F3` produces.
So a rule that hunts those pairs has to be *shown* to accept correct accented
text, in the languages that put those characters after an accented letter, or
it is a rule that will be switched off. The clean twin cannot carry that
evidence: it is three lines of ASCII-plus-three-characters, and the failure
being excluded is not non-ASCII, it is non-ASCII in the right arrangement.

There is a third file here that is neither damage nor a control: a
**directory** holding a known number of tripping files, which is the witness
for the skip-budget comparison. It is built by this builder rather than by the
gate for the same reason every other fixture is: a gate that wrote its own
witness would have two places to forget it, and one of them would be the file
doing the forgetting.

That last one is the one most likely to be left out. A non-UTF-8 file is not
automatically damage: a Latin-1 test fixture is a test *input*, and a gate that
flags it teaches its users to ignore it. The gate's answer is a declared
exemption list -- and a declared list nobody ever exercises is a permanent hole
wearing a justification. So this builder also produces a Latin-1 file, and the
gate must honour the exemption for it. The mutation checks prove the hatches are
shut; the Latin-1 file proves the escape hatch opens.

# Which signatures, and why those two mojibake rules

The predicates are the ones this repository's own content makes reachable, not a
general library of encoding checks:

* **BOM** -- what `Set-Content -Encoding UTF8` and `Out-File` write on Windows
  PowerShell 5.1. Valid UTF-8, so a decode-only check cannot see it.
* **Truncated sequence** -- a write interrupted mid-character.
* **U+FFFD** -- a lossy decode that was then saved.
* **CJK then `?`** -- a byte GBK could not represent, lost on a Chinese-Windows
  round trip, with the loss made visible.
* **`U+00E2 U+20AC`** -- a `U+2xxx` character (em dash, en dash, curly quote)
  re-read as
  cp1252. The em dash alone appears in 13 of this tree's 16 Markdown files and in
  the CLI help strings.
* **`U+00C3 U+2026`** -- `U+00C5`, the Angstrom sign, re-read as cp1252. This is
  the most common non-ASCII character in the Python, and it is the one a Western
  PowerShell pair mangles.
* **A losslessly-substituted cp936 character** -- `U+811C` sitting in the file
  where `U+00C5` was. One of the two that destroys **nothing**: no
  byte is lost, no `?` is written, no `U+FFFD` appears, the file decodes as
  perfectly good UTF-8, and every one of the six byte-level rules above returns
  `None` on it. It is here because it was found in this tree *after* the gate had
  been reporting a clean run, three of them in one Rust source file, in angstrom
  positions.
* **A cp936 image of a three-byte character** -- a UTF-8 sequence longer than two
  bytes cannot survive a cp936 read at all, so this is the second flavour of
  `nothing destroyed, no trace left`, except that it garbles its neighbour as
  well because the trailing byte is joined to whatever came next.
* **A cp936 image of a four-byte character** -- the same class one width up, and
  the only signature in this file that this tree cannot produce on its own: its
  sources are supplementary-plane characters and 0 of 318 source files hold one.
  It is here precisely because it cannot fire here.
* **A two-character cp1252 image of an accented letter** -- the tenth fixture and
  the ninth predicate, and the one the two rules above were a stand-in for. A
  measurement over the cp1252 table found **2,891** reachable two-character
  images, of which the two older rules named exactly two: the image of a dash and
  the image of an Angstrom sign. The image of `U+00E9` -- the modal mojibake on
  earth, and what a Western PowerShell pair does to every accented letter in a
  file -- was not among them, and neither was the image of any other accented
  character. So the tenth fixture is the modal case and the ninth predicate reads
  the whole table instead of two rows of it.

**The cost of the two older rules was stated, and this round it was paid.** The
paragraph above used to end "a different accented character misread by cp1252 is
not caught", which was true and was the whole problem: the two rules were narrow
on purpose, to avoid firing on French and Portuguese text, and the price was that
the common case was invisible. The ninth rule keeps the narrowness where it is
needed -- six second characters are excluded because correct text demonstrably
puts them after an accented letter, and this repository's own `angstrom` plus
superscript three is 69 occurrences of one of them -- and reads the other 2,885
outright. The cost that remains is stated rather than hidden: a file whose only
accented damage produces one of those six is not caught, and the gate's own
self-test builds exactly that file and asserts it is the one case that stays
green.

**Those two sequences are named by code point here, never written out.** Spelling
them literally would put the mojibake in this file, and this file lives in the
tree `check_text_encoding.py` scans -- so the gate would go red on the builder of
its own fixtures, naming the very signatures the builder exists to produce. The
same trap caught the previous version of the gate over its own `U+FFFD`
constant, and it is named in that file's docstring for the same reason. The
seventh fixture needs the same discipline for the same reason, and for a second
one besides: `U+811C` is a perfectly ordinary CJK ideograph, so there is no
escaping of it that a reader could mistake for prose.

**A note on the escapes as a rule rather than as a habit.** This file escapes
every character it hunts, and so does the gate, and so should anything else in
this tree that needs to *name* a damage signature rather than contain one. The
reason is not tidiness and it is not an exemption: a literal character and the
same character written as a `\\uXXXX` escape are different byte strings with the
same runtime value, so the escape costs a test nothing and leaves the hunted
bytes out of the source text entirely. That is what lets the gate scan this
file, and `docs/`, and a `cli_check.py` oracle asserting that a decode did not
happen -- all at full width, with no list of files allowed to hold a specimen.
`check_text_encoding.py`'s `_not_damage` is the same sentence said to a reader
at the moment they need it.

The paragraph above had to be written twice. Its first version quoted the
literal character it was explaining, which put the hunted bytes into the file
that hunts them -- and the file caught it, because it is scanned like everything
else. That is the entire argument in one incident, and it is worth recording
that the mistake survived a careful draft and was caught by a measurement rather
than by reading.

# The damage is declared, not accidental

`<out>/DAMAGE.txt` states in prose *and* in a `damage.*` machine-readable block
what each file is and what was done to it. The failure this is written against is
the obvious one: write some bytes, call the result a fixture, and leave the next
reader unable to tell a deliberate mutation from a file somebody corrupted for
real.

# Where it writes, and what it will not do

`--out` is required, and the gate passes a `tempfile.mkdtemp()` path, so the
damage lands under `%TEMP%` and never inside the repository. This script writes
**only** inside the directory it is given and it deletes nothing. A fixture built
by damaging a real file in place would be a defect rather than a test, and this
repository has a recorded history of files being damaged in place by exactly the
mechanism under test.

# Determinism

The same flags produce byte-identical files, `DAMAGE.txt` included: nothing
written here is a timestamp, a hostname or a path outside `--out`. Two fixtures
differing is therefore a difference in the tree, not in when they were built.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DAMAGE_NAME = "DAMAGE.txt"
CLEAN_NAME = "clean_baseline.py"
LATIN1_NAME = "latin1_fixture.txt"

#: The clean twin. It carries **non-ASCII on purpose** -- an em dash, an Angstrom
#: sign and a CJK character -- because a negative control made of ASCII would
#: satisfy "accepts the clean file" against a gate whose predicates had all been
#: switched off: a predicate hunting non-ASCII damage cannot fire on bytes that
#: have none. Every character here is one this repository really uses. `--` and
#: `U+00C5` are in `cli.py`'s help strings and `U+4E2D` is in
#: `dock-core/src/grid.rs`.
CLEAN_SOURCE = (
    "# The clean twin of the ten mutations beside it.\n"
    "# Angstrom: \u00c5. Dash: \u2014. CJK: \u4e2d.\n"
    "RADIUS = 8.0  # \u00c5ngstr\u00f6m, per-atom \u2014 not a charge\n"
)

#: A clean Chinese sentence followed by the cp936 mojibake of one word in it.
#:
#: **The eighth signature, and the one the gate's own docstring used to describe
#: without measuring.** A UTF-8 sequence longer than two bytes cannot survive a
#: cp936 read -- the codec takes two bytes per character for every lead byte in
#: `0x81`-`0xFE` -- so `E6 96 87` is read as one character from `(E6, 96)` and
#: its trailing `87` is joined to the first byte of whatever came next. The old
#: text called that "the loud flavour" and pointed at no rule for it. It is
#: enumerable, it is caught, and this is the file that proves it.
#:
#: **Both halves are in this one file, and that is the whole point.** The
#: predicate keys on the image *and* on a character this image could have come
#: from, because a forward rule cannot work: the images are ordinary ideographs,
#: which is what a Chinese file is supposed to contain. The damage consumes the
#: copy of the source that was in the file, so the clean character has to
#: survive *somewhere else* -- and it does, in the first half. A fixture
#: carrying only the mojibake would prove nothing, which is a limit of the
#: class rather than of the fixture, and the predicate's own message says so.
#:
#: Escaped, never written literally, for the reason the seventh fixture records:
#: this file lives in the tree `check_text_encoding.py` scans, so a literal
#: mojibake ideograph here would make the gate red on the builder of its own
#: fixtures. The image of `U+6587` is `U+93C2`.
MULTIBYTE_SOURCE = (
    "# \u8fd9\u662f\u6587\u4ef6\u8bf4\u660e\uff0c\u5199\u5f97\u6b63\u786e\u3002\n"
    "# Pasted from a source that was already mojibake: \u93c2\u56e6\u6b22\n"
)

#: The same class, the **four**-byte width, and the reason this is a separate
#: fixture rather than a second sentence in the one above.
#:
#: `GBK_MULTIBYTE` has two halves that fail for different reasons and are
#: reachable by different sources. The three-byte half's sources are
#: `U+0800`-`U+FFFF`; the four-byte half's are **supplementary-plane only**,
#: `U+10000`-`U+10FFFF`, all 1,048,576 of them, because that is the only way a
#: UTF-8 sequence reaches four bytes. Measured over every source tree in this
#: repository with build output pruned, **0 of 318 files carry a single
#: supplementary-plane character**, and 0 carry a four-byte-half image, so that
#: half cannot fire on this tree no matter what else changes -- it is not "0 on
#: a small sample", it is "0 because the only characters that can produce a hit
#: are absent from the corpus".
#:
#: **A code path that cannot fire is still worth shipping, and this is why.**
#: The class is a fact about the codec rather than about this repository, and the
#: first source file carrying an emoji gives it a member; a rule deleted while it
#: is quiet has to be re-derived from a bug report. But *shipped* is not the same
#: as *exercised*, and this builder's doctrine is one fixture per signature, so
#: the honest form of "keep it" is a fixture that goes red if the four-byte
#: branch stops working -- which is this one. Without it the four-byte half
#: would be the one rule here with no oracle, and a fixture with no oracle is
#: not a fixture.
#:
#: Escaped for the same reason as every other character in this file, and with
#: `\U` rather than `\u` because a supplementary-plane code point does not fit in
#: four hex digits. `U+10000` encodes as `F0 90 80 80` and cp936 reads the
#: leading pair as `U+995C`; the two trailing bytes are joined to whatever
#: follows, which is the neighbour damage the predicate's message describes.
MULTIBYTE_4BYTE_SOURCE = (
    "# A supplementary-plane character: \U00010000, written correctly.\n"
    "# Pasted from a source that was already mojibake: \u995c\n"
)

#: Correct accented prose, which **every predicate must accept**. The second
#: negative control, and the only one that speaks the language the new cp1252
#: rule is about. It lives beside `CLEAN_SOURCE` rather than further down
#: because the M10 mutation is built from it, and a module-level dict literal
#: is evaluated in order.
#:
#: **Every shape in it was chosen because a correct file really contains it,
#: and four of them are this repository's own text.** A cp1252 image of one
#: character is two characters, and a rule keyed on the pair cannot tell the
#: difference between "a damaged `U+00F3`" and this project's own cubic
#: angstrom -- they are the same two characters. So the file carries:
#:
#: * `U+00C5 U+00B3` -- angstrom-cubed, 52 times across 10 real files here as
#:   measured on 2026-10-03;
#: * `U+00C5 U+00B2` -- angstrom-squared, in `odck.md` and `docs/SCORING.md`;
#: * `U+00C5 U+2014` -- angstrom then the Chinese em dash, which that
#:   convention writes with no space, as `README.md` and `docs/VERIFICATION.md`
#:   both do;
#: * `U+00D7 U+00B2` -- a cross product squared, from
#:   `dock-core/src/kinematics.rs`;
#: * a no-break space after an accented letter, which is French and Spanish
#:   typography and which the cp936 rules have never had to think about;
#: * Czech words where a caron follows an accented letter, which is ordinary
#:   and which no amount of reading this gate's code would have suggested.
#:
#: A rule that refuses any of these is not strict, it is wrong about physics
#: and about four languages, and it would be switched off inside a week. This
#: file is what stops that happening silently. **It is also the base of the
#: M10 mutation**, so the damaged fixture and this one differ by exactly one
#: word and the pair is a controlled experiment rather than two files that
#: happen to be near each other.
ACCENTED_PROSE_NAME = "correct_accented_prose.txt"

#: The one word the M10 fixture damages, held as an escape so this file holds
#: no specimen of the damage it exists to build. `U+00E9` is the modal
#: mojibake source: read as cp1252 its own two UTF-8 bytes become two
#: characters, and nothing is lost, so the file stays valid UTF-8 and reads as
#: correct-but-wrong.
LATIN1_SOURCE = "caf\u00e9"

ACCENTED_PROSE = (
    "# Correct accented text, which every predicate must accept.\n"
    "# Angstrom units: 140 \u00c5\u00b2, a 88 \u00c5\u00b3 groove, "
    "12 \u00c5\u00b3 of space.\n"
    "# The cross product squared: [\u03b8]\u00d7\u00b2, and [\u03b8]\u00d7.\n"
    "# The Chinese em dash convention, no space before it: 10.53 "
    "\u00c5\u2014\u2014 monotonic.\n"
    "# French: le caf\u00e9 est pr\u00eat, cr\u00e9er n\u2019est pas voir, "
    "dit : oui, c\u00f4t\u00e9 \u00a0: le go\u00fbt, une "
    "note\u2014en passant\u2014reste, cr\u00e9at-il\u00a0? "
    "C\u2019est l\u2019\u00e9cole de No\u0153l et "
    "l\u0153uvre, c\u0153ur, b\u0153uf. "
    "Prix \u00a0: 12\u20ac. 3\u00b2, 1\u00bc2, 5\u00b5s, "
    "1\u00b2, 2\u00ba, 3\u00aa, 1\u00b9, 3\u00bc, "
    "1\u00bd, 3\u00be4, \u00b0, caf\u00e9\u2122.\n"
    "# All-caps French, which is where a capital-range accented letter -- the "
    "# only kind of character a COMPLETE cp1252 image can start with -- is "
    "followed by a no-break space and by a soft hyphen:\n"
    "# MANTE\u0101QUE CR\u00c9\u00a0: non. CR\u00c9\u00adATION "
    "D\u00c9FINITIVE.\n"
    "# Portuguese: S\u00e3o Paulo \u00e9 a maior cidade, "
    "Ant\u00f4nia n\u00e3o vem \u00e1qui, "
    "cora\u00e7\u00e3o, fam\u00edlia, d\u2019\u00e1gua, "
    "n\u00e3o, \u00e1, \u00e3o, \u00f5es, Jo\u00e3o, "
    "O caf\u00e9 \u00e9 bom\u00bb de verdade.\n"
    "# Spanish: el a\u00f1o es el coraz\u00f3n, Sebasti\u00e1n "
    "dijo \u00bfqu\u00e9 haces?\u00a0\u00a1Venga! M\u00e1s de "
    "3\u00b2 a\u00f1os, 2\u00bc de sol, el ni\u00f1o y la "
    "monta\u00f1a\u00e1n aqu\u00ed, Se\u00f1or Ca\u00f1iz.\n"
    "# Czech: De\u0161\u0165\u00e9 den p\u0159\u00e1\u017elo. "
    "P\u00e1\u017e je tu, je\u0161te, v\u00e1\u017e\u00e1m, "
    "po\u0161li, n\u00e1m co d\u011blat, s\u00ed\u00f1, "
    "\u0160estn\u00ed, \u010d\u00e9st, ne\u017e\u00e1p\u00ed, "
    "U\u0159ek je za\u010d\u00e1tkou.\n"
    "# All-caps Czech, where a capital-range accented letter is followed by "
    "# S-caron: the ordinary word for height, and the only correct text in "
    "# this file that can produce a COMPLETE image with S-caron in it.\n"
    "# M\u011a\u0160 V\u00dd\u0160KA A TU\u0160, "
    "\u0160\u00c9\u0160T\u00cd Z\u011a\u010c.\n"
    "# Nordic and German: \u00c5ngstr\u00f6m i "
    "r\u00f6nt, Gr\u00f6\u00df Gott, Wei\u00df und "
    "Sch\u00e4tze, Ma\u00df \u00fcber alles, "
    "B\u00fccher k\u00f6nnen, fr\u00fch\u00e5hlig "
    "\u00e4rgern, \u00d8\u00f8re, \u00e6\u00f8\u00e5, "
    "fl\u00ea\u0192ne.\n"
    "# CJK, so the cp936 rules are exercised by the same file: "
    "\u8fd9\u662f\u6587\u4ef6\u8bf4\u660e\uff0c"
    "\u5199\u5f97\u6b63\u786e\u3002\n"
)

#: One entry per signature: `(tag, predicate, bytes, what was done and why it is
#: the signature that matters)`.
#:
#: Every mutation replaces exactly one occurrence and leaves the other copies of
#: the same character intact, so a fixture is a *partially* damaged file rather
#: than a wholly different one -- which is closer to what a real round trip
#: produces, and means a predicate that refused the file for the wrong reason
#: would still have had to explain itself.
MUTATIONS = {
    "bom_damaged.py": (
        "M1", "bom",
        b"\xef\xbb\xbf" + CLEAN_SOURCE.encode("utf-8"),
        "a UTF-8 BOM in front of bytes that are otherwise clean: exactly what "
        "PowerShell 5.1's `Set-Content -Encoding UTF8` and `Out-File` write, and "
        "a valid UTF-8 file besides, so a decode-only check cannot see it",
    ),
    "truncated_sequence.py": (
        "M2", "undecodable",
        CLEAN_SOURCE.encode("utf-8").replace(b"\xc3\x85", b"\xc3", 1),
        "the second byte of one two-byte sequence removed, so the file ends on "
        "a lead byte with nothing after it: a write that was interrupted, or an "
        "editor that saved a partial buffer",
    ),
    "replacement_char.py": (
        "M3", "replacement",
        CLEAN_SOURCE.encode("utf-8").replace(b"\xc3\x85", b"\xef\xbf\xbd", 1),
        "U+FFFD where the Angstrom sign was. The file decodes cleanly and is "
        "wrong: something decoded it with `errors=replace` and saved the result",
    ),
    "lost_byte_gbk.py": (
        "M4", "lost_byte",
        CLEAN_SOURCE.encode("utf-8")
        .replace(b"\xc3\x85", "\u4e2d?".encode("utf-8"), 1),
        "a CJK character immediately followed by `?`: a byte GBK could not "
        "represent, lost on a Chinese-Windows round trip, with the loss visible",
    ),
    "cp1252_dash.py": (
        "M5", "mojibake_dash",
        CLEAN_SOURCE.encode("utf-8").replace(
            b"\xe2\x80\x94", "\u00e2\u20ac\u201d".encode("utf-8"), 1),
        "an em dash re-read as cp1252 and written back, which is what a Western "
        "PowerShell 5.1 `Get-Content -Raw` / `Set-Content -Encoding UTF8` pair "
        "does to every typographic character in a file",
    ),
    "cp1252_angstrom.py": (
        "M6", "mojibake_angstrom",
        CLEAN_SOURCE.encode("utf-8").replace(
            b"\xc3\x85", "\u00c3\u2026".encode("utf-8"), 1),
        "an Angstrom sign re-read as cp1252 and written back. The most common "
        "non-ASCII character in this tree's Python, and the one a Western "
        "PowerShell pair mangles most often",
    ),
    "lossless_gbk.py": (
        "M7", "lossless_gbk",
        CLEAN_SOURCE.encode("utf-8").replace(
            b"\xc3\x85", "\u811c".encode("utf-8"), 1),
        "one Angstrom sign re-read as cp936 and written back, with the other "
        "one left correct -- which is what makes this the only fixture of the "
        "seven that destroys nothing at all. The file decodes as valid UTF-8, "
        "no byte was lost, no `?` and no U+FFFD were written, and all six "
        "byte-level rules return None on it. `U+00C5` is `C3 85`, and a "
        "Chinese-Windows `Get-Content -Raw` reads that pair as one good GBK "
        "character. Both halves of the signature are in this one file on "
        "purpose: the correct sign is what tells mojibake from a rare Chinese "
        "character, so a fixture carrying only the substituted one would prove "
        "half the rule",
    ),
    "multibyte_gbk.py": (
        "M8", "multibyte_gbk",
        CLEAN_SOURCE.encode("utf-8") + MULTIBYTE_SOURCE.encode("utf-8"),
        "one word of a Chinese sentence left in cp936 mojibake beside the same "
        "word written correctly. A UTF-8 sequence longer than two bytes cannot "
        "survive a cp936 read at all, so this class destroys nothing and "
        "silently garbles its neighbour, and none of the six byte-level rules "
        "above can see it. The file decodes as valid UTF-8, carries no BOM, no "
        "`?` and no U+FFFD. Both halves are here on purpose: the damage consumed "
        "the copy of the clean character that was in the file, so the predicate "
        "has nothing to pair the image with unless the correct character "
        "survives somewhere else, and that is what the first line supplies",
    ),
    "multibyte_gbk_4byte.py": (
        "M9", "multibyte_gbk",
        CLEAN_SOURCE.encode("utf-8") + MULTIBYTE_4BYTE_SOURCE.encode("utf-8"),
        "the four-byte half of the same class, and the half this tree cannot "
        "produce on its own. `U+10000` is four UTF-8 bytes; cp936 takes two per "
        "character, so it reads as `U+995C` and the remaining two bytes are "
        "joined to whatever follows -- the neighbour damage again, and again "
        "with no byte lost and no `?` or U+FFFD written, which is why all six "
        "byte-level rules above return None on it. The clean character is on "
        "the line above for the same reason the three-byte fixture keeps its "
        "own: the damage consumed the copy that was here, so the predicate has "
        "nothing to pair the image with unless the source survives elsewhere. "
        "This fixture exists so the four-byte branch has an oracle at all: the "
        "0 hits this predicate reports on a clean tree is a fact about this "
        "repository holding no supplementary-plane character, and not a fact "
        "about the branch working",
    ),
    "cp1252_e_acute.py": (
        "M10", "mojibake_cp1252",
        ACCENTED_PROSE.encode("utf-8").replace(
            # The source word and its image, both derived rather than typed.
            # `LATIN1_SOURCE` is an escape so that this file carries no
            # specimen of the damage it exists to build -- the same rule every
            # other fixture here follows, and the reason the image is computed
            # by reading the source's own bytes as cp1252 instead of written
            # out. The gate would go red on its own fixture builder otherwise,
            # and it would be red for the right reason.
            LATIN1_SOURCE.encode("utf-8"),
            LATIN1_SOURCE.encode("utf-8").decode("cp1252").encode("utf-8"), 1),
        "the modal mojibake on earth, and the reason the other two cp1252 "
        "fixtures were not enough. `U+00E9` is `C3 A9`, and a Western "
        "PowerShell 5.1 `Get-Content -Raw` reads those two bytes as two "
        "characters and `Set-Content -Encoding UTF8` writes them back, so the "
        "word is still readable and still wrong. Nothing about the file is "
        "malformed: it decodes as UTF-8, no byte is lost, no `?` and no U+FFFD "
        "is written, and the two older cp1252 rules do not fire on it because "
        "they name a dash and an Angstrom sign and not the image of an "
        "accented letter. A measurement over the cp1252 table found 2,891 "
        "reachable two-character images and this rule named two of them, so "
        "this fixture is the case that was invisible. **The bytes are derived, "
        "not typed**: the replacement is computed by reading the source "
        "character's own UTF-8 as cp1252, which is the same derivation the "
        "predicate reads its table from, so this file cannot drift into "
        "carrying a specimen that is not the damage it claims to be",
    ),
}

#: The correct-accented-prose file and the Latin-1 file are the two negative
#: controls, and they are different in kind. The Latin-1 one proves the
#: exemption path *opens*; this one proves a rule about two-character mojibake
#: does not fire on correct accented text, which is the failure mode that
#: would switch the rule off.
NEGATIVE_CONTROLS = (CLEAN_NAME, ACCENTED_PROSE_NAME, LATIN1_NAME)

#: The name of the directory the budget fixture lives in. **Not** a member of
#: the gate's `SKIP_DIRS`, and that is deliberate: `iter_files` prunes by name,
#: so a fixture directory called `dist` would be pruned by the very walk that
#: is supposed to measure it, and the budget arithmetic would be tested
#: against an empty directory forever.
SKIP_FIXTURE_DIR = "od-skip-cost-fixture"
#: How many files in it trip a predicate, and how many do not. **These two
#: numbers are the budget's witness**, and the gate measures them rather than
#: being told them: see the self-test.
SKIP_FIXTURE_HITS = 3
SKIP_FIXTURE_CLEAN = 2

#: Deliberately Latin-1, and therefore not valid UTF-8. **The gate must accept
#: this one.** It is a test input rather than damage, and the only reason a gate
#: can tell the difference is that the difference is *declared*: this path appears
#: in the gate's `ENCODING_EXEMPT` with a reason, for the duration of the
#: self-test only.
LATIN1_BYTES = b"# A deliberately Latin-1 test input: caf\xe9 na\xefve r\xe9sum\xe9.\n"


def manifest() -> str:
    """The `DAMAGE.txt` body: prose first, then the machine-readable block.

    The `damage.<name>=<tag>` lines are what the gate matches its own predicates
    against; `damage.why.<name>` is the prose, and it is written out rather than
    held in the gate so that the two cannot drift into telling different stories
    about the same file.
    """
    lines = [
        "Deliberate encoding damage, built by damaged_encoding_fixture.py.",
        "",
        "Every file here is a FIXTURE. None of it is source and none of it is a",
        "copy of a real file. The clean twin's bytes are written from scratch and",
        "each mutation is that same byte string with one documented edit applied,",
        "so nothing here was produced by damaging anything: the point is to give",
        "the gate something to be wrong about, not to simulate an accident.",
        "",
        "damage.manifest=%s" % DAMAGE_NAME,
        "damage.clean=%s  # the negative control: every predicate must accept it"
        % CLEAN_NAME,
        "damage.accented=%s  # the accented negative control: correct text in "
        "six languages plus this repository's own physics notation, which "
        "every predicate must also accept" % ACCENTED_PROSE_NAME,
        "damage.exempt=%s  # declared Latin-1: must be exempt, not red"
        % LATIN1_NAME,
        "damage.skip_fixture=%s  # the budget witness: a directory holding "
        "%d tripping and %d clean files, so the gate can measure a cost of "
        "known size and prove the budget comparison in both directions"
        % (SKIP_FIXTURE_DIR, SKIP_FIXTURE_HITS, SKIP_FIXTURE_CLEAN),
    ]
    for name, (tag, predicate, _data, why) in sorted(MUTATIONS.items()):
        # `M1 bom` -- the tag, then the predicate's *short* name. Both are
        # machine-readable on purpose: the gate matches the second field against
        # the predicates it actually has, so a fixture cannot be declared against
        # a rule that has since been renamed or removed.
        lines.append("damage.%s=%s %s" % (name, tag, predicate))
        lines.append("damage.why.%s=%s" % (name, why))
    return "\n".join(lines) + "\n"


def build_skip_fixture(root: Path) -> Path:
    """The budget witness: a directory whose damage cost is known exactly.

    **Why the number of tripping files is the witness, and not a file that
    "proves" the budget.** A budget is an arithmetic claim, and an arithmetic
    claim is falsifiable only when both sides of the comparison are produced
    together. The right-hand side is a number in the gate's own source, which
    never disappears; the left-hand side is a property of a real directory
    that moves with every build. So the only way to show the comparison still
    bites is to hand `measure_skip` a directory whose cost this builder knows,
    and let the gate *measure* it rather than be told it. The files are real
    damaged files, so the cost that comes back is the cost the real directories
    produce, and `SKIP_FIXTURE_HITS` is an expectation to be checked rather
    than an input to the check -- the self-test red if the two disagree, which
    is what would happen if a predicate started or stopped firing on them.
    """
    where = root / SKIP_FIXTURE_DIR
    where.mkdir(parents=True, exist_ok=True)
    for n in range(1, SKIP_FIXTURE_HITS + 1):
        (where / f"damaged_{n}.py").write_bytes(
            MUTATIONS["cp1252_e_acute.py"][2])
    for n in range(1, SKIP_FIXTURE_CLEAN + 1):
        (where / f"clean_{n}.py").write_bytes(
            CLEAN_SOURCE.encode("utf-8"))
    return where


def build(out: Path) -> int:
    """Write the fixture set into `out`. Returns a process exit code."""
    out.mkdir(parents=True, exist_ok=True)
    (out / CLEAN_NAME).write_bytes(CLEAN_SOURCE.encode("utf-8"))
    (out / ACCENTED_PROSE_NAME).write_bytes(ACCENTED_PROSE.encode("utf-8"))
    (out / LATIN1_NAME).write_bytes(LATIN1_BYTES)
    for name, (_tag, _pred, data, _why) in sorted(MUTATIONS.items()):
        (out / name).write_bytes(data)
    skip_where = build_skip_fixture(out)
    (out / DAMAGE_NAME).write_text(manifest(), encoding="utf-8")

    written = [CLEAN_NAME, ACCENTED_PROSE_NAME, LATIN1_NAME,
               *sorted(MUTATIONS), DAMAGE_NAME]
    print(f"wrote {len(written)} file(s) and 1 directory into {out}")
    for name in written:
        data = (out / name).read_bytes()
        print(f"  {name:24s} {len(data):6d} B  first bytes {data[:12]!r}")
    kids = sorted(p.name for p in skip_where.iterdir())
    print(f"  {SKIP_FIXTURE_DIR + '/':24s} {len(kids):6d} file(s)  {kids}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="directory to write the fixtures into; must not be "
                         "inside the repository")
    args = ap.parse_args()
    return build(Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
