"""
Reading speed experiment: measures maximum comprehensible reading speed
for different text presentations. Text is flashed at a given WPM, then
a comprehension question appears. Use SPACE to begin each trial, mouse
to select answers.

Logging:
- Saves everything needed to replicate each trial (text, question, options, correct, wpm, ms/word, etc.)
- Saves reaction time (RT) from when the question is FIRST rendered to click.
- Saves experiment config + RNG seed so the exact trial sequence can be reproduced.
"""

import json
import math
import random
import time

import pygame
import pygame.freetype
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ============== Configuration ==============
CONFIG = {
    "questions_file": "questions.txt",
    "n_trials": 6,
    "wpm_range": (150, 600),  # min and max WPM to sample (upper bound exclusive)
    "ms_between_words": None,  # override ms/word; None => 60000/wpm
    "screen_width": 900,
    "screen_height": 700,
    "bg_color": (230, 230, 245),
    "text_color": (30, 30, 35) ,
    "accent_color": (100, 180, 255),
    "font_size_text": 24,
    "font_size_question": 22,
    "font_size_options": 20,
    "line_spacing": 1.4,
    "option_padding": 12,
    "inter_trial_blank_ms": 500,
    # logging + reproducibility
    "log_dir": "logs",
    "rng_seed": None,  # set to an int to reproduce exactly; None => auto-generate and log it
}


# ============== Data structures ==============
@dataclass
class QuestionItem:
    text: str
    question: str
    options: list[str]         # ["A) ...", "B) ...", ...]
    correct_answer: str        # "A", "B", "C", or "D"


@dataclass
class Trial:
    question_item: QuestionItem
    wpm: int
    bold_proportion: tuple[float, float] | None = None


# ============== Parser ==============
def load_questions(filepath: str) -> list[QuestionItem]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {filepath}")

    raw = path.read_text(encoding="utf-8")
    items: list[QuestionItem] = []
    blocks = raw.split("###")

    for block in blocks:
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
            line = lines[i]
            if line.strip() == "TEXT":
                i += 1
                while i < len(lines) and lines[i].strip() not in ("Q", "QUESTION"):
                    text_lines.append(lines[i].strip())
                    i += 1
                continue
            if line.strip() in ("Q", "QUESTION"):
                i += 1
                if i < len(lines):
                    question = lines[i].strip()
                    i += 1
                continue
            if line.strip() == "OPTS":
                i += 1
                while i < len(lines) and lines[i].strip() not in ("ANS", "ANSWER"):
                    opt = lines[i].strip()
                    if opt:
                        options.append(opt)
                    i += 1
                continue
            if line.strip() in ("ANS", "ANSWER"):
                i += 1
                if i < len(lines):
                    answer = lines[i].strip().upper()[0]
                    i += 1
                continue
            i += 1

        text = " ".join(text_lines).strip()
        if text and question and options and answer:
            items.append(
                QuestionItem(
                    text=text,
                    question=question,
                    options=options,
                    correct_answer=answer,
                )
            )

    return items


class Experiment:
    def __init__(self, questions: list[QuestionItem], n_trials: int, wpm_range: tuple[int, int], config: dict):
        self.questions = questions
        self.n_trials = n_trials
        self.wpm_range = wpm_range
        self.config = config

        # reproducible RNG for trial sampling
        self.seed = self.config.get("rng_seed")
        if self.seed is None:
            self.seed = random.randrange(0, 2**32)
        self.rng = random.Random(self.seed)

        self.trials = self.sample_trials()

        self.font_text = pygame.freetype.SysFont("Arial", config["font_size_text"])
        try:
            self.font_text_bold = pygame.freetype.SysFont("Arial Bold", config["font_size_text"])
        except (OSError, ValueError):
            self.font_text_bold = self.font_text  # fallback: no bold on this system
        self.font_question = pygame.freetype.SysFont("Arial", config["font_size_question"])
        self.font_options = pygame.freetype.SysFont("Arial", config["font_size_options"])
        for f in (self.font_text, self.font_text_bold, self.font_question, self.font_options):
            f.origin = True
        self.M_ADV_X = 4  # index of horizontal_advance_x in get_metrics()

        self.screen = pygame.display.set_mode((config["screen_width"], config["screen_height"]))
        pygame.display.set_caption("Reading Speed Experiment")
        self.clock = pygame.time.Clock()

        self.results: list[dict] = []
        self.trial_idx = 0
        self.state = "ready"  # "ready" | "showing_text" | "showing_question"

        self.text_start_time_s = 0.0
        self.option_rects: list[tuple[pygame.Rect, str]] = []  # (rect, answer_letter)

        # Reaction time: question onset (first render time)
        self.question_start_time_s: float | None = None

        # Logging
        self.log_path = self._init_logfile()

    def sample_trials(self) -> list[Trial]:
        if self.n_trials > len(self.questions):
            raise ValueError(f"n_trials={self.n_trials} exceeds available questions={len(self.questions)}")

        chosen_questions = self.rng.sample(self.questions, self.n_trials)
        chosen_wpms = self.rng.sample(range(self.wpm_range[0], self.wpm_range[1]), self.n_trials)
        return [Trial(q, w, bold_proportion=(0.0, 0.3)) for q, w in zip(chosen_questions, chosen_wpms)]

    def ms_per_word_for_trial(self, wpm: int) -> float:
        """Milliseconds each word is displayed. Uses CONFIG override if set."""
        if self.config.get("ms_between_words") is not None:
            return float(self.config["ms_between_words"])
        return 60000.0 / wpm if wpm > 0 else 0.0

    def draw_centered_word(
        self,
        surf: pygame.Surface,
        word: str,
        word_start_char: int,
        bold_proportion: tuple[float, float] | None,
    ):
        """Draw a single word; characters in bold_proportion range are bold. Center the display on the bolded section."""
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
        bold_start, bold_end = bold_proportion if bold_proportion else (-1, -1)
        # Character indices in this word that fall in bold range (passage indices)
        word_bold_start = max(0, math.floor(bold_start * len(word)))
        word_bold_end = min(len(word), math.ceil(bold_end * len(word)))
        has_bold_in_word = word_bold_start < word_bold_end
        x = 0
        x_positions = [0]  # x at start of each character
        for i, letter in enumerate(word):
            use_bold = word_bold_start <= i < word_bold_end
            f = font_bold if use_bold else font
            metric = (metrics_bold if use_bold else metrics_reg)[i]
            f.render_to(text_surf, (x, baseline), letter, color)
            x += metric[self.M_ADV_X]
            x_positions.append(x)
        # Position: baseline at fixed y; horizontal position from percentage (0=left, 50=center, 100=right)
        screen_rect = surf.get_rect()
        baseline_y = screen_rect.centery        
        text_rect = text_surf.get_rect()
        text_rect.y = int(baseline_y - baseline)
        if has_bold_in_word:
            bold_x_start = x_positions[word_bold_start]
            bold_x_end = x_positions[word_bold_end]
            bold_center_x = (bold_x_start + bold_x_end) / 2
            text_rect.x = int(screen_rect.centerx - bold_center_x)
        else:
            text_rect.x = int(screen_rect.centerx - text_rect.width / 2)
        surf.blit(text_surf, text_rect)

    # ---------- logging helpers ----------
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
            "n_trials": self.n_trials,
            "wpm_range": self.wpm_range,
            "pygame_version": pygame.version.ver,
        }
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def _log(self, record: dict):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------- main loop ----------
    def run(self):
        running = True
        while running and self.trial_idx < len(self.trials):
            trial = self.trials[self.trial_idx]
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if self.state == "ready" and event.key == pygame.K_SPACE:
                        self.state = "showing_text"
                        self.text_start_time_s = pygame.time.get_ticks() / 1000.0
                        self.question_start_time_s = None  # reset for this trial

                        # log trial start (replication-critical)
                        words = trial.question_item.text.split()
                        word_ms = self.ms_per_word_for_trial(trial.wpm)
                        self._log({
                            "record_type": "trial_start",
                            "timestamp": datetime.now().isoformat(),
                            "trial_index": self.trial_idx,
                            "trial_number": self.trial_idx + 1,
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
                    pos = event.pos
                    for rect, letter in self.option_rects:
                        if rect.collidepoint(pos):
                            click_time_s = pygame.time.get_ticks() / 1000.0
                            # question_start_time_s is set when question is first rendered
                            rt_ms = None
                            if self.question_start_time_s is not None:
                                rt_ms = (click_time_s - self.question_start_time_s) * 1000.0

                            correct = letter == trial.question_item.correct_answer

                            trial_record = {
                                "trial": self.trial_idx + 1,
                                "trial_index": self.trial_idx,
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
                            }
                            self.results.append(trial_record)

                            # log response
                            self._log({
                                "record_type": "trial_response",
                                "timestamp": datetime.now().isoformat(),
                                **trial_record,
                            })

                            self.trial_idx += 1
                            self.state = "ready"
                            self.option_rects = []
                            self.question_start_time_s = None
                            break

            # ----- draw -----
            self.screen.fill(self.config["bg_color"])
            # Red vertical line down the center
            # cx = self.screen.get_width() // 2
            # pygame.draw.line(
            #     self.screen, (255, 0, 0), (cx, 0), (cx, self.screen.get_height()), 2
            # )

            if self.state == "ready":
                ms = self.ms_per_word_for_trial(trial.wpm)
                prompt = f"Trial {self.trial_idx + 1} of {len(self.trials)}  —  {int(ms)} ms/word"
                prompt_surf, prompt_rect = self.font_question.render(
                    prompt, fgcolor=self.config["accent_color"]
                )
                prompt_rect.midtop = (self.screen.get_width() // 2, 80)
                self.screen.blit(prompt_surf, prompt_rect)
                instruct = "Press SPACE to begin"
                inst_surf, inst_rect = self.font_text.render(
                    instruct, fgcolor=self.config["text_color"]
                )
                inst_rect.midtop = (self.screen.get_width() // 2, 300)
                self.screen.blit(inst_surf, inst_rect)

            elif self.state == "showing_text":
                elapsed_ms = (pygame.time.get_ticks() / 1000.0 - self.text_start_time_s) * 1000.0
                word_ms = self.ms_per_word_for_trial(trial.wpm)
                words = trial.question_item.text.split()
                total_duration_ms = len(words) * word_ms

                if elapsed_ms < total_duration_ms:
                    word_idx = int(elapsed_ms / word_ms) if word_ms > 0 else 0
                    if 0 <= word_idx < len(words):
                        word_start_char = sum(len(words[j]) + 1 for j in range(word_idx))
                        self.draw_centered_word(
                            self.screen,
                            words[word_idx],
                            word_start_char,
                            trial.bold_proportion,
                        )
                else:
                    # brief blank before question
                    self.state = "showing_question"
                    self.option_rects = []
                    self.screen.fill(self.config["bg_color"])
                    pygame.display.flip()
                    pygame.time.delay(self.config["inter_trial_blank_ms"])
                    # DO NOT set question_start_time here; set on first render

            elif self.state == "showing_question":
                # set question start time on first frame the question is actually rendered
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
                q_rect.topleft = (50, 80)
                self.screen.blit(q_surf, q_rect)

                self.option_rects = []
                y = 180
                for opt in q.options:
                    letter = opt[0].upper() if opt else ""
                    opt_surf, opt_rect = self.font_options.render(
                        opt, fgcolor=self.config["text_color"]
                    )
                    pad = self.config["option_padding"]
                    w = max(400, opt_rect.width + pad * 2)
                    h = opt_rect.height + pad * 2
                    x = (self.screen.get_width() - w) // 2
                    rect = pygame.Rect(x, y, w, h)

                    pygame.draw.rect(self.screen, self.config["bg_color"], rect, border_radius=8)
                    pygame.draw.rect(self.screen, self.config["accent_color"], rect, 1, border_radius=8)
                    self.screen.blit(opt_surf, (x + pad, y + pad))

                    self.option_rects.append((rect, letter))
                    y += h + 10

            pygame.display.flip()

        pygame.quit()

        # Print results
        print("\n=== Results ===")
        for r in self.results:
            status = "OK" if r["correct"] else "X"
            ms = int(r["ms_per_word"])
            rt = "NA" if r["rt_ms"] is None else f"{int(r['rt_ms'])} ms"
            print(f"Trial {r['trial']}: {r['wpm']} WPM ({ms} ms/word) — {status} — RT {rt}")

        correct_count = sum(1 for r in self.results if r["correct"])
        print(f"\nScore: {correct_count}/{len(self.results)} correct")
        print(f"Log saved to: {self.log_path}")

        # log footer
        self._log({
            "record_type": "experiment_footer",
            "timestamp": datetime.now().isoformat(),
            "total_trials_completed": len(self.results),
            "num_correct": correct_count,
        })


if __name__ == "__main__":
    pygame.init()
    questions = load_questions(CONFIG["questions_file"])
    experiment = Experiment(
        questions=questions,
        n_trials=CONFIG["n_trials"],
        wpm_range=CONFIG["wpm_range"],
        config=CONFIG,
    )
    experiment.run()