"""
Reading speed experiment: measures maximum comprehensible reading speed
for different proportions of bolded text (e.g. bionic reading).

Bayesian adaptive procedure
────────────────────────────
One independent Bayesian adaptive estimator runs per condition (bold proportion).

The psychometric function models P(correct | WPM, threshold):

    P = guess_rate + (1 - lapse_rate - guess_rate) × Φ((log t − log WPM) / σ)

where Φ is the standard normal CDF, t is the threshold WPM, and σ controls
the slope.  The function is 1 at very low WPM and falls to guess_rate at high WPM.
The threshold t is the WPM at the midpoint between guess_rate and (1 − lapse_rate).

A prior p(t) is maintained over a fine log-WPM grid.  After each trial the
posterior is updated via Bayes' theorem.  The next test WPM is set to the
posterior-mean threshold, always placing the next trial at the current best
estimate.  The estimator stops when the posterior SD in log-WPM falls below
convergence_sd (i.e., the threshold is localised to within ~±convergence_sd
log-WPM, roughly ±(convergence_sd × 100)% in WPM) or max_trials is reached.

Conditions are interleaved in round-robin order.

Logging
───────
JSONL file per session in log_dir/.  Records:
  • experiment_header  – config, seed, per-condition prior setup
  • trial_start        – full trial metadata
  • question_shown     – question onset timestamp (RT anchor)
  • trial_response     – answer, RT, posterior snapshot after update
  • experiment_footer  – final threshold estimates, CIs, per-condition stats
"""

import json
import math
import random
import time

import pygame
import pygame.freetype
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ══════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════

CONFIG = {
    "questions_file": "questions.txt",
    
    # HARD CODED PASSAGE / QUESTION VALUES
    "num_passages": 6,
    "questions_per_passage": 6,   
    "shuffle_passages": True,

    # ── Conditions (one Bayesian estimator each) ──────────────────────────
    # bold_proportion = [start_frac, end_frac] of each word's characters to bold.
    # None = no bolding (baseline).  [0.0, 0.5] = first 50 % of every word.
    "conditions": [
#        {"id": "no_bold", "label": "No bold (baseline)", "bold_proportion": None},
        {"id": "bio_start",  "label": "Bionic start 30 %",        "bold_proportion": [0.0, 0.30]},
        {"id": "bio_middle",  "label": "Bionic middle 30 %",        "bold_proportion": [0.35, 0.65]},
        # {"id": "bio_end",  "label": "Bionic end 30 %",        "bold_proportion": [0.7, 1.0]},
    ],

    # ── Bayesian estimator parameters ─────────────────────────────────────
    # Psychometric function
    "guess_rate": 0.25,     # P(correct) at very high WPM (= 1 / n_answer_choices)
    "lapse_rate": 0.04,     # P(incorrect) at very easy WPM (inattention / typos)
    "sigma": 0.35,          # psychometric slope width in log-WPM units;
                            #   0.35 ≈ the function spans ~1.4× WPM range
    # Prior over log-threshold
    "initial_wpm": 350,     # centre of the Gaussian prior
    "prior_log_sigma": 0.7, # prior width (0.7 ≈ ±2× WPM 68 % CI around initial)
    "wpm_min": 80,
    "wpm_max": 1000,
    "n_grid": 300,          # posterior grid resolution

    # Stopping
    "max_trials_per_condition": 12,   # hard cap (exact number of trials per condition)
    "convergence_sd": 0.05,              # stop when posterior SD (log-WPM) < this

    # ── Display ───────────────────────────────────────────────────────────
    "screen_width": 900,
    "screen_height": 700,
    "bg_color": (230, 230, 245),
    "text_color": (30, 30, 35),
    "accent_color": (100, 180, 255),
    "correct_color": (60, 160, 80),
    "incorrect_color": (200, 60, 60),
    "font_size_text": 28,
    "font_size_question": 22,
    "font_size_options": 20,
    "font_size_small": 16,
    "line_spacing": 1.4,
    "option_padding": 12,
    "inter_trial_blank_ms": 500,
    "show_progress": False,
    "round_robin": False,


    # ── Logging / reproducibility ─────────────────────────────────────────
    "log_dir": "logs",
    "rng_seed": None,            # int to reproduce exactly; None = auto-generate

    # ── Override ──────────────────────────────────────────────────────────
    "ms_between_words": None,    # ms/word override; None = derive from WPM
}


# ══════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════
PASSAGE_TO_QUESTION = {}
@dataclass
class QuestionItem:
    # TODO: Discriminate between inference and recall questions
    # type: str 
    text: str
    question: str
    options: list[str]       # ["A) ...", "B) ...", ...]
    correct_answer: str      # "A", "B", "C", or "D"
    passage_id: int = 0
    within_passage_idx: int = 0


@dataclass
class Trial:
    question_item: QuestionItem
    wpm: int
    bold_proportion: tuple[float, float] | None = None
    condition_id: str = ""


@dataclass
class BayesianAdaptiveState:
    """
    QUEST-inspired Bayesian adaptive threshold estimator for one condition.

    Maintains a posterior p(t) over a log-WPM grid.  After each trial the
    posterior is updated via Bayes.  The next test WPM is the posterior mean
    of the threshold (always testing at the best current estimate).

    Stopping: halts when posterior SD < convergence_sd OR n_trials >= max_trials.
    """

    condition_id: str
    label: str
    bold_proportion: tuple[float, float] | None
    wpm_min: int
    wpm_max: int
    initial_wpm: int
    prior_log_sigma: float   # Gaussian prior width in log-WPM
    guess_rate: float        # floor of psychometric function (chance)
    lapse_rate: float        # gap below ceiling (inattention)
    sigma: float             # psychometric slope width in log-WPM
    n_grid: int
    max_trials: int
    convergence_sd: float    # stop when posterior SD (log-WPM) < this

    # ── running state (with defaults so dataclass ordering is valid) ──────
    n_trials: int = 0
    n_correct: int = 0
    is_complete: bool = False
    _next_wpm: int = 0
    _prev_mean_log: float = 0.0
    _log_grid: list = field(default_factory=list)
    _posterior: list = field(default_factory=list)
    _history: list = field(default_factory=list)

    def __post_init__(self):
        log_min = math.log(self.wpm_min)
        log_max = math.log(self.wpm_max)
        self._log_grid = [
            log_min + (log_max - log_min) * i / (self.n_grid - 1)
            for i in range(self.n_grid)
        ]
        # Gaussian prior centred on initial_wpm
        log_init = math.log(self.initial_wpm)
        prior = [
            math.exp(-0.5 * ((g - log_init) / self.prior_log_sigma) ** 2)
            for g in self._log_grid
        ]
        total = sum(prior)
        self._posterior = [p / total for p in prior]
        self._next_wpm = self._posterior_mean_wpm()
        self._prev_mean_log = math.log(max(1, self._next_wpm))

    # ── psychometric function ─────────────────────────────────────────────

    def _p_correct(self, log_wpm: float, log_t: float) -> float:
        """P(correct | WPM, threshold t).  Decreasing in WPM."""
        z = (log_t - log_wpm) / self.sigma
        return self.guess_rate + (1.0 - self.lapse_rate - self.guess_rate) * _norm_cdf(z)

    # ── Bayes update ──────────────────────────────────────────────────────

    def update(self, correct: bool) -> tuple[str, bool]:
        """
        Update posterior given the outcome of a trial at self._next_wpm.
        Returns (direction, False) where direction ∈ {"up","down","none"} describes
        the shift in the threshold estimate (useful for logging).
        """
        tested_log_wpm = math.log(max(1, self._next_wpm))

        new_post = [
            self._posterior[i] * (
                self._p_correct(tested_log_wpm, log_t) if correct
                else (1.0 - self._p_correct(tested_log_wpm, log_t))
            )
            for i, log_t in enumerate(self._log_grid)
        ]
        total = sum(new_post)
        if total > 1e-300:
            self._posterior = [p / total for p in new_post]

        self.n_trials += 1
        if correct:
            self.n_correct += 1

        new_mean_log = math.log(max(1, self._posterior_mean_wpm()))
        direction = (
            "up"   if new_mean_log > self._prev_mean_log + 0.01 else
            "down" if new_mean_log < self._prev_mean_log - 0.01 else
            "none"
        )
        self._prev_mean_log = new_mean_log
        self._next_wpm = self._posterior_mean_wpm()

        self._history.append({
            "trial": self.n_trials,
            "tested_wpm": int(round(math.exp(tested_log_wpm))),
            "correct": correct,
            "threshold_est_wpm": round(math.exp(new_mean_log), 1),
            "posterior_sd_log": round(self.posterior_sd_log, 4),
        })

        sd = self.posterior_sd_log
        if self.n_trials >= self.max_trials or (
                self.n_trials >= 6 and sd < self.convergence_sd):
            self.is_complete = True

        return direction, False   # False = no "reversal" concept in Bayesian

    # ── posterior statistics ──────────────────────────────────────────────

    def _posterior_mean_wpm(self) -> int:
        mean_log = sum(g * p for g, p in zip(self._log_grid, self._posterior))
        return max(self.wpm_min, min(self.wpm_max, int(round(math.exp(mean_log)))))

    @property
    def posterior_sd_log(self) -> float:
        """Posterior SD in log-WPM units."""
        mean_log = sum(g * p for g, p in zip(self._log_grid, self._posterior))
        var = sum((g - mean_log) ** 2 * p
                  for g, p in zip(self._log_grid, self._posterior))
        return math.sqrt(max(0.0, var))

    @property
    def current_wpm(self) -> int:
        return self._next_wpm

    @property
    def threshold_estimate(self) -> float | None:
        if self.n_trials < 3:
            return None
        return float(self._posterior_mean_wpm())

    @property
    def threshold_95ci(self) -> tuple[float, float] | None:
        """95 % credible interval for the threshold WPM."""
        if self.n_trials < 3:
            return None
        cumsum = 0.0
        lo = math.exp(self._log_grid[0])
        hi = math.exp(self._log_grid[-1])
        found_lo = False
        for g, p in zip(self._log_grid, self._posterior):
            cumsum += p
            if not found_lo and cumsum >= 0.025:
                lo = math.exp(g)
                found_lo = True
            if cumsum >= 0.975:
                hi = math.exp(g)
                break
        return lo, hi

    # ── serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        est = self.threshold_estimate
        ci = self.threshold_95ci
        acc = round(self.n_correct / self.n_trials * 100, 1) if self.n_trials else None
        return {
            "condition_id": self.condition_id,
            "label": self.label,
            "bold_proportion": list(self.bold_proportion) if self.bold_proportion else None,
            "current_wpm": self.current_wpm,
            "n_trials": self.n_trials,
            "n_correct": self.n_correct,
            "pct_correct": acc,
            "threshold_estimate_wpm": round(est, 1) if est is not None else None,
            "threshold_95ci_wpm": [round(ci[0], 1), round(ci[1], 1)] if ci else None,
            "posterior_sd_log": round(self.posterior_sd_log, 4),
            "trial_history": self._history,
            "is_complete": self.is_complete,
        }


# ══════════════════════════════════════════════════════════════════
#  Questions parser
# ══════════════════════════════════════════════════════════════════

def load_questions(filepath: str, questions_per_passage: int = 4) -> list[QuestionItem]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {filepath}")

    raw = path.read_text(encoding="utf-8")
    items: list[QuestionItem] = []

    # Each "###" block is one (text subsection + question + options + answer)
    for block in raw.split("###"):
        block = block.strip()
        if not block:
            continue

        lines = block.split("\n")
        text_lines: list[str] = []
        question = ""
        options: list[str] = []
        answer = ""

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == "TEXT":
                i += 1
                while i < len(lines) and lines[i].strip() not in ("Q", "QUESTION"):
                    s = lines[i].strip()
                    if s:
                        text_lines.append(s)
                    i += 1
                continue

            if line in ("Q", "QUESTION"):
                i += 1
                if i < len(lines):
                    question = lines[i].strip()
                    i += 1
                continue

            if line == "OPTS":
                i += 1
                while i < len(lines) and lines[i].strip() not in ("ANS", "ANSWER"):
                    opt = lines[i].strip()
                    if opt:
                        options.append(opt)
                    i += 1
                continue

            if line in ("ANS", "ANSWER"):
                i += 1
                if i < len(lines) and lines[i].strip():
                    answer = lines[i].strip().upper()[0]
                i += 1
                continue

            i += 1

        text = " ".join(text_lines).strip()
        if not (text and question and options and answer):
            continue

        item_idx = len(items)
        passage_id = item_idx // questions_per_passage
        within_idx = item_idx % questions_per_passage

        qi = QuestionItem(
            text=text,
            question=question,
            options=options,
            correct_answer=answer,
            passage_id=passage_id,
            within_passage_idx=within_idx,
        )
        items.append(qi)

    # Optional: build a proper mapping passage_id -> list[QuestionItem]
    PASSAGE_TO_QUESTION.clear()
    for qi in items:
        PASSAGE_TO_QUESTION.setdefault(qi.passage_id, []).append(qi)

    print("DEBUG passage groups:", {k: len(v) for k, v in PASSAGE_TO_QUESTION.items()})
    return items


# ══════════════════════════════════════════════════════════════════
#  Experiment
# ══════════════════════════════════════════════════════════════════

class Experiment:
    def __init__(self, questions: list[QuestionItem], config: dict):
        self.questions = questions
        self.config = config

        # ── RNG ───────────────────────────────────────────────────────────
        self.seed: int = config.get("rng_seed") or random.randrange(0, 2**32)
        self.rng = random.Random(self.seed)

        # ── Question pool ─────────────────────────────────────────────────
        self.passage_to_items: dict[int, list[QuestionItem]] = {}
        for qi in questions:
            self.passage_to_items.setdefault(qi.passage_id, []).append(qi)

        # ensure within-passage order is stable (0,1,2,3)
        for pid in self.passage_to_items:
            self.passage_to_items[pid].sort(key=lambda x: x.within_passage_idx)

        self.passage_ids = sorted(self.passage_to_items.keys())
        if self.config.get("shuffle_passages", True):
            self.rng.shuffle(self.passage_ids)

        self._passage_cursor = 0
        self._within_cursor = 0

        # ── Bayesian estimators (one per condition) ────────────────────────
        self.staircases: dict[str, BayesianAdaptiveState] = {}
        self._sc_order: list[str] = []
        self._sc_cursor = 0
        self._build_estimators()

        # ── Fonts ─────────────────────────────────────────────────────────
        self.font_text = pygame.freetype.SysFont(
            "Arial",
            config["font_size_text"],
            bold=False
        )

        self.font_text_bold = pygame.freetype.SysFont(
            "Arial",
            config["font_size_text"],
            bold=True
        )

        self.font_question = pygame.freetype.SysFont("Arial", config["font_size_question"])
        self.font_options = pygame.freetype.SysFont("Arial", config["font_size_options"])
        self.font_small = pygame.freetype.SysFont("Arial", config["font_size_small"])

        for f in (
            self.font_text,
            self.font_text_bold,
            self.font_question,
            self.font_options,
            self.font_small,
        ):
            f.origin = True

        self.M_ADV_X = 4

        # ── Display ───────────────────────────────────────────────────────
        self.screen = pygame.display.set_mode(
            (config["screen_width"], config["screen_height"])
        )
        pygame.display.set_caption("Reading Speed Experiment")
        self.clock = pygame.time.Clock()

        # ── State machine ─────────────────────────────────────────────────
        self.results: list[dict] = []
        self.trial_idx = 0
        self.state = "ready"
        self.text_start_time_s = 0.0
        self.option_rects: list[tuple[pygame.Rect, str]] = []
        self.question_start_time_s: float | None = None

        self.current_trial: Trial | None = self._next_trial()

        # ── Logging ───────────────────────────────────────────────────────
        self.log_path = self._init_logfile()

    # ── estimator helpers ─────────────────────────────────────────────────

    def _build_estimators(self):
        for c in self.config.get("conditions", []):
            bp = c.get("bold_proportion")
            if bp is not None:
                bp = tuple(bp)
            est = BayesianAdaptiveState(
                condition_id=c["id"],
                label=c.get("label", c["id"]),
                bold_proportion=bp,
                wpm_min=self.config.get("wpm_min", 80),
                wpm_max=self.config.get("wpm_max", 1000),
                initial_wpm=self.config.get("initial_wpm", 250),
                prior_log_sigma=self.config.get("prior_log_sigma", 0.7),
                guess_rate=self.config.get("guess_rate", 0.25),
                lapse_rate=self.config.get("lapse_rate", 0.04),
                sigma=self.config.get("sigma", 0.35),
                n_grid=self.config.get("n_grid", 300),
                max_trials=self.config.get("max_trials_per_condition", 25),
                convergence_sd=self.config.get("convergence_sd", 0.08),
            )
            self.staircases[c["id"]] = est
            self._sc_order.append(c["id"])

    def _next_trial(self) -> Trial | None:
        if self.config["round_robin"]:
            for sc_id in self._sc_order:
                sc = self.staircases[sc_id]
                if not sc.is_complete:
                    return Trial(
                        question_item=self._pick_question(),
                        wpm=sc.current_wpm,
                        bold_proportion=sc.bold_proportion,
                        condition_id=sc_id,
                    )
            return None
        else:
            # complete all estimators for one condition at a time


            for sc_id in self._sc_order:
                sc = self.staircases[sc_id]
                if not sc.is_complete:
                    break
            else:
                return None
            return Trial(
                question_item=self._pick_question(),
                wpm=sc.current_wpm,
                bold_proportion=sc.bold_proportion,
                condition_id=sc_id,
            )

    def _pick_question(self) -> QuestionItem:
        if self._passage_cursor >= len(self.passage_ids):
            # fallback (should not happen unless you run out)
            return self.questions[self.rng.randrange(len(self.questions))]

        pid = self.passage_ids[self._passage_cursor]
        items = self.passage_to_items[pid]

        q = items[self._within_cursor]
        self._within_cursor += 1

        # if we finished the 4 questions for this passage, advance to next passage
        if self._within_cursor >= len(items):
            self._passage_cursor += 1
            self._within_cursor = 0

        return q

    # ── per-word ms ───────────────────────────────────────────────────────

    def ms_per_word_for_trial(self, wpm: int) -> float:
        if self.config.get("ms_between_words") is not None:
            return float(self.config["ms_between_words"])
        return 60000.0 / wpm if wpm > 0 else 0.0

    # ── word rendering ────────────────────────────────────────────────────

    def draw_centered_word(
        self,
        surf: pygame.Surface,
        word: str,
        word_start_char: int,
        bold_proportion: tuple[float, float] | None,
    ):
        """Render word centered on screen; bold the [start,end) fraction of characters."""
        if not word:
            return
        color = self.config["text_color"]
        font = self.font_text
        font_bold = self.font_text_bold
        rect = font.get_rect(word)
        rect_bold = font_bold.get_rect(word)
        w = max(rect.width, rect_bold.width)
        h = max(rect.height, rect_bold.height)
        text_surf = pygame.Surface((w, h))
        text_surf.fill(self.config["bg_color"])
        baseline = rect.y
        metrics_reg = font.get_metrics(word)
        metrics_bold = font_bold.get_metrics(word)
        bold_start, bold_end = bold_proportion if bold_proportion else (-1.0, -1.0)
        word_bold_start = max(0, math.floor(bold_start * len(word)))
        word_bold_end = min(len(word), math.ceil(bold_end * len(word)))
        has_bold = word_bold_start < word_bold_end
        x = 0
        x_positions = [0]
        for i, letter in enumerate(word):
            use_bold = word_bold_start <= i < word_bold_end
            f = font_bold if use_bold else font
            metric = (metrics_bold if use_bold else metrics_reg)[i]
            f.render_to(text_surf, (x, baseline), letter, color)
            x += metric[self.M_ADV_X]
            x_positions.append(x)
        screen_rect = surf.get_rect()
        text_rect = text_surf.get_rect()
        text_rect.y = int(screen_rect.centery - baseline)
        
        if has_bold:
            bold_center_x = (x_positions[word_bold_start] + x_positions[word_bold_end]) / 2
            text_rect.x = int(screen_rect.centerx - bold_center_x)
        else:
            text_rect.x = int(screen_rect.centerx - text_rect.width / 2)
        
        # DO NOT CENTER WHEN MIDDLE BOLD 
        # text_rect.x = int(screen_rect.centerx - text_rect.width / 2)
        surf.blit(text_surf, text_rect)

    # ── logging ───────────────────────────────────────────────────────────

    def _init_logfile(self) -> Path:
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"reading_speed_{ts}.jsonl"
        header = {
            "record_type": "experiment_header",
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "config": self.config,
            "estimators": [sc.to_dict() for sc in self.staircases.values()],
            "pygame_version": pygame.version.ver,
        }
        path.write_text(json.dumps(header, default=list, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        return path

    def _log(self, record: dict):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=list, ensure_ascii=False) + "\n")

    # ── drawing helpers ───────────────────────────────────────────────────

    def _blit_centered(self, font: pygame.freetype.Font, text: str,
                       color: tuple, y: int) -> pygame.Rect:
        surf, rect = font.render(text, fgcolor=color)
        rect.midtop = (self.screen.get_width() // 2, y)
        self.screen.blit(surf, rect)
        return rect

    def _draw_progress_panel(self, highlight_id: str | None, top_y: int) -> int:
        """
        Draw per-condition Bayesian progress bars.
        Bar fill = n_trials / max_trials.
        Label shows current threshold estimate and posterior SD.
        Returns y below the last row.
        """
        cw = self.screen.get_width()
        bar_w = 280
        bar_h = 8
        x0 = (cw - bar_w) // 2
        y = top_y

        for sc in self.staircases.values():
            is_active = sc.condition_id == highlight_id
            color = self.config["accent_color"] if is_active else self.config["text_color"]

            # condition label + arrow
            prefix = "▶ " if is_active else "  "
            label_surf, label_rect = self.font_small.render(
                f"{prefix}{sc.label}", fgcolor=color
            )
            label_rect.topleft = (x0, y)
            self.screen.blit(label_surf, label_rect)
            y += 20

            # progress bar
            fill = min(1.0, sc.n_trials / sc.max_trials)
            pygame.draw.rect(self.screen, (190, 190, 210), (x0, y, bar_w, bar_h), border_radius=4)
            if fill > 0:
                pygame.draw.rect(self.screen, color,
                                 (x0, y, int(bar_w * fill), bar_h), border_radius=4)

            # posterior info right of bar
            est = sc.threshold_estimate
            if est is not None:
                # express uncertainty in WPM: ≈ est × sd_log
                sd_wpm = int(round(est * sc.posterior_sd_log))
                ci = sc.threshold_95ci
                ci_str = f"  95% CI [{round(ci[0])}–{round(ci[1])}]" if ci else ""
                info = f"{round(est)} ±{sd_wpm} WPM{ci_str}  ({sc.n_trials}/{sc.max_trials})"
            else:
                info = f"(collecting data… {sc.n_trials}/{sc.max_trials})"
            if sc.is_complete:
                info += "  ✓"

            info_surf, info_rect = self.font_small.render(info, fgcolor=color)
            info_rect.midleft = (x0 + bar_w + 10, y + bar_h // 2)
            self.screen.blit(info_surf, info_rect)

            y += bar_h + 16

        return y

    # ── main loop ─────────────────────────────────────────────────────────

    def run(self):
        running = True

        while running:
            self.clock.tick(60)

            # ── events ────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if self.state == "end" and event.key == pygame.K_q:
                        running = False
                        break
                    if self.state == "ready" and event.key == pygame.K_SPACE:
                        if self.current_trial is None:
                            self.state = "end"
                            break
                        trial = self.current_trial
                        self.state = "showing_text"
                        self.text_start_time_s = pygame.time.get_ticks() / 1000.0
                        self.question_start_time_s = None
                        words = trial.question_item.text.split()
                        word_ms = self.ms_per_word_for_trial(trial.wpm)
                        self._log({
                            "record_type": "trial_start",
                            "timestamp": datetime.now().isoformat(),
                            "trial_number": self.trial_idx + 1,
                            "trial_index": self.trial_idx,
                            "condition_id": trial.condition_id,
                            "condition_label": self.staircases[trial.condition_id].label,
                            "bold_proportion": list(trial.bold_proportion) if trial.bold_proportion else None,
                            "wpm": trial.wpm,
                            "ms_per_word": word_ms,
                            "num_words": len(words),
                            "planned_text_duration_ms": len(words) * word_ms,
                            "text": trial.question_item.text,
                            "question": trial.question_item.question,
                            "options": trial.question_item.options,
                            "correct_answer": trial.question_item.correct_answer,
                        })

                if event.type == pygame.MOUSEBUTTONDOWN and self.state == "showing_question":
                    trial = self.current_trial
                    if trial is None:
                        continue
                    for rect, letter in self.option_rects:
                        if rect.collidepoint(event.pos):
                            click_time_s = pygame.time.get_ticks() / 1000.0
                            rt_ms = None
                            if self.question_start_time_s is not None:
                                rt_ms = (click_time_s - self.question_start_time_s) * 1000.0

                            correct = letter == trial.question_item.correct_answer

                            sc = self.staircases[trial.condition_id]
                            direction, _ = sc.update(correct)
                            ci = sc.threshold_95ci

                            trial_record = {
                                "record_type": "trial_response",
                                "timestamp": datetime.now().isoformat(),
                                "trial_number": self.trial_idx + 1,
                                "trial_index": self.trial_idx,
                                "condition_id": trial.condition_id,
                                "condition_label": sc.label,
                                "bold_proportion": list(trial.bold_proportion) if trial.bold_proportion else None,
                                "wpm": trial.wpm,
                                "ms_per_word": self.ms_per_word_for_trial(trial.wpm),
                                "num_words": len(trial.question_item.text.split()),
                                "text": trial.question_item.text,
                                "question": trial.question_item.question,
                                "options": trial.question_item.options,
                                "correct_answer": trial.question_item.correct_answer,
                                "chosen_answer": letter,
                                "correct": correct,
                                "rt_ms": rt_ms,
                                # posterior snapshot after update
                                "bayes_direction": direction,
                                "bayes_next_wpm": sc.current_wpm,
                                "bayes_threshold_est": (
                                    round(sc.threshold_estimate, 1)
                                    if sc.threshold_estimate is not None else None
                                ),
                                "bayes_threshold_95ci": (
                                    [round(ci[0], 1), round(ci[1], 1)] if ci else None
                                ),
                                "bayes_posterior_sd_log": round(sc.posterior_sd_log, 4),
                                "bayes_n_trials_done": sc.n_trials,
                                "bayes_is_complete": sc.is_complete,
                                "bayes_snapshot": sc.to_dict(),
                            }
                            self.results.append(trial_record)
                            self._log(trial_record)

                            self.trial_idx += 1
                            self.option_rects = []
                            self.question_start_time_s = None
                            self.current_trial = self._next_trial()
                            self.state = "end" if self.current_trial is None else "ready"
                            break

            # ── draw ──────────────────────────────────────────────────────
            self.screen.fill(self.config["bg_color"])

            if self.state == "ready":
                trial = self.current_trial
                if trial is None:
                    self.state = "end"
                else:
                    sc = self.staircases[trial.condition_id]
                    ms = self.ms_per_word_for_trial(trial.wpm)
                    if self.config["show_progress"]:
                        self._blit_centered(
                            self.font_question,
                            f"Trial {self.trial_idx + 1}",
                            self.config["text_color"], 30
                        )
                        self._blit_centered(
                            self.font_question,
                            f"{trial.wpm} WPM  ·  {int(ms)} ms/word",
                            self.config["accent_color"], 60
                        )
                        bar_bottom = self._draw_progress_panel(trial.condition_id, 108)
                    # fixation cross
                    cx = self.screen.get_width() // 2
                    cy = self.screen.get_height() // 2
                    arm = 16
                    thickness = 2
                    cross_color = self.config["text_color"]
                    pygame.draw.line(self.screen, cross_color,
                                     (cx - arm, cy), (cx + arm, cy), thickness)
                    pygame.draw.line(self.screen, cross_color,
                                     (cx, cy - arm), (cx, cy + arm), thickness)

                    self._blit_centered(
                        self.font_text, "Press SPACE to begin",
                        self.config["text_color"], 200
                    )

            elif self.state == "showing_text":
                trial = self.current_trial
                if trial is None:
                    self.state = "end"
                else:
                    elapsed_ms = (pygame.time.get_ticks() / 1000.0 - self.text_start_time_s) * 1000.0
                    word_ms = self.ms_per_word_for_trial(trial.wpm)
                    words = trial.question_item.text.split()
                    total_ms = len(words) * word_ms

                    if elapsed_ms < total_ms:
                        word_idx = int(elapsed_ms / word_ms) if word_ms > 0 else 0
                        if 0 <= word_idx < len(words):
                            word_start_char = sum(len(words[j]) + 1 for j in range(word_idx))
                            self.draw_centered_word(
                                self.screen, words[word_idx],
                                word_start_char, trial.bold_proportion
                            )
                    else:
                        self.state = "showing_question"
                        self.option_rects = []
                        self.screen.fill(self.config["bg_color"])
                        pygame.display.flip()
                        pygame.time.delay(self.config["inter_trial_blank_ms"])

            elif self.state == "showing_question":
                trial = self.current_trial
                if trial is None:
                    self.state = "end"
                else:
                    if self.question_start_time_s is None:
                        self.question_start_time_s = pygame.time.get_ticks() / 1000.0
                        self._log({
                            "record_type": "question_shown",
                            "timestamp": datetime.now().isoformat(),
                            "trial_index": self.trial_idx,
                            "trial_number": self.trial_idx + 1,
                            "question_start_time_s": self.question_start_time_s,
                        })

                    q = trial.question_item
                    q_surf, q_rect = self.font_question.render(
                        q.question, fgcolor=self.config["text_color"]
                    )
                    q_rect.topleft = (50, 40)
                    self.screen.blit(q_surf, q_rect)

                    self.option_rects = []
                    y = 120
                    for opt in q.options:
                        letter = opt[0].upper() if opt else ""
                        opt_surf, opt_rect = self.font_options.render(
                            opt, fgcolor=self.config["text_color"]
                        )
                        pad = self.config["option_padding"]
                        w = max(500, opt_rect.width + pad * 2)
                        h = opt_rect.height + pad * 2
                        x = (self.screen.get_width() - w) // 2
                        rect = pygame.Rect(x, y, w, h)
                        pygame.draw.rect(self.screen, self.config["bg_color"], rect, border_radius=8)
                        pygame.draw.rect(self.screen, self.config["accent_color"], rect, 1, border_radius=8)
                        self.screen.blit(opt_surf, (x + pad, y + pad))
                        self.option_rects.append((rect, letter))
                        y += h + 8

            elif self.state == "end":
                self._draw_end_screen()

            pygame.display.flip()

        pygame.quit()
        self._print_results()
        self._log({
            "record_type": "experiment_footer",
            "timestamp": datetime.now().isoformat(),
            "total_trials_completed": len(self.results),
            "num_correct": sum(1 for r in self.results if r["correct"]),
            "threshold_estimates": {
                sc_id: sc.to_dict() for sc_id, sc in self.staircases.items()
            },
        })

    # ── end screen ────────────────────────────────────────────────────────

    def _draw_end_screen(self):
        cw = self.screen.get_width()
        self._blit_centered(
            self.font_question, "Experiment complete!",
            self.config["accent_color"], 36
        )

        # column x positions
        col = {"label": 55, "thresh": 340, "ci": 460, "acc": 650, "sd": 750}
        header_y = 96

        def small(text, x, y, color=None):
            color = color or self.config["text_color"]
            surf, rect = self.font_small.render(text, fgcolor=color)
            rect.topleft = (x, y)
            self.screen.blit(surf, rect)

        accent = self.config["accent_color"]
        small("Condition",      col["label"],  header_y, accent)
        small("Threshold",      col["thresh"], header_y, accent)
        small("95 % CI",        col["ci"],     header_y, accent)
        small("Accuracy",       col["acc"],    header_y, accent)
        small("SD (log)",       col["sd"],     header_y, accent)

        sep_y = header_y + 22
        pygame.draw.line(self.screen, accent, (col["label"], sep_y), (cw - 40, sep_y), 1)

        y = sep_y + 10
        for sc in self.staircases.values():
            est = sc.threshold_estimate
            ci = sc.threshold_95ci
            color = (self.config["correct_color"] if sc.is_complete
                     else self.config["text_color"])

            thresh_str = f"{round(est)} WPM" if est is not None else "N/A"
            ci_str = (f"[{round(ci[0])}–{round(ci[1])}]"
                      if ci is not None else "N/A")
            acc_str = (f"{sc.n_correct}/{sc.n_trials} "
                       f"({round(sc.n_correct/sc.n_trials*100) if sc.n_trials else 0} %)")
            sd_str = f"{sc.posterior_sd_log:.3f}"

            small(sc.label,    col["label"],  y, color)
            small(thresh_str,  col["thresh"], y, color)
            small(ci_str,      col["ci"],     y, color)
            small(acc_str,     col["acc"],    y, color)
            small(sd_str,      col["sd"],     y, color)
            y += 34

        self._blit_centered(
            self.font_small, f"Log: {self.log_path}",
            self.config["text_color"], y + 18
        )
        self._blit_centered(
            self.font_small, "Press Q or close window to quit",
            self.config["text_color"], y + 40
        )

    # ── terminal summary ──────────────────────────────────────────────────

    def _print_results(self):
        print("\n=== Results ===\n")
        hdr = (f"{'Condition':<24} {'Threshold':>10} {'95 % CI':>18} "
               f"{'Accuracy':>12} {'Trials':>8} {'SD(log)':>9}")
        print(hdr)
        print("─" * len(hdr))
        for sc in self.staircases.values():
            est = sc.threshold_estimate
            ci = sc.threshold_95ci
            thresh_str = f"{round(est)} WPM" if est is not None else "N/A"
            ci_str = (f"[{round(ci[0])}–{round(ci[1])}] WPM"
                      if ci is not None else "N/A")
            acc_str = (f"{sc.n_correct}/{sc.n_trials} "
                       f"({round(sc.n_correct/sc.n_trials*100) if sc.n_trials else 0} %)")
            sd_str = f"{sc.posterior_sd_log:.3f}"
            print(f"{sc.label:<24} {thresh_str:>10} {ci_str:>18} "
                  f"{acc_str:>12} {sc.n_trials:>8} {sd_str:>9}")
        print(f"\nLog saved to: {self.log_path}")


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pygame.init()
    questions = load_questions(CONFIG["questions_file"], questions_per_passage=CONFIG["questions_per_passage"])
    experiment = Experiment(questions=questions, config=CONFIG)
    experiment.run()
