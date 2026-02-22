"""
Reading speed experiment: measures maximum comprehensible reading speed
for different text presentations. Text is flashed at a given WPM, then
a comprehension question appears. Use SPACE to begin each trial, mouse
to select answers.
"""

import pygame

import random
from dataclasses import dataclass
from pathlib import Path


# ============== Configuration ==============
CONFIG = {
    "questions_file": "questions.txt",
    "n_trials": 6,
    "wpm_range": (150, 600),  # min and max WPM to sample
    "ms_between_words": None,  # ms each word is shown; None = derive from trial WPM (60000/wpm)
    "screen_width": 900,
    "screen_height": 700,
    "bg_color": (30, 30, 35),
    "text_color": (240, 240, 245),
    "accent_color": (100, 180, 255),
    "font_size_text": 24,
    "font_size_question": 22,
    "font_size_options": 20,
    "line_spacing": 1.4,
    "option_padding": 12,
    "inter_trial_blank_ms": 500,
}


# ============== Data structures ==============
@dataclass
class QuestionItem:
    text: str
    question: str
    options: list[str]  # ["A) ...", "B) ...", ...]
    correct_answer: str  # "A", "B", "C", or "D"


@dataclass
class Trial:
    question_item: QuestionItem
    wpm: int


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

def word_count(text: str) -> int:
    return len(text.split())

class Experiment:
    def __init__(self, questions: list[QuestionItem], n_trials: int, wpm_range: tuple[int, int], config: dict):
        self.questions = questions
        self.n_trials = n_trials
        self.wpm_range = wpm_range
        self.trials = self.sample_trials()
        self.config = config
        self.font_text = pygame.font.SysFont("Arial", config["font_size_text"])
        self.font_question = pygame.font.SysFont("Arial", config["font_size_question"])
        self.font_options = pygame.font.SysFont("Arial", config["font_size_options"])
        self.screen = pygame.display.set_mode((config["screen_width"], config["screen_height"]))
        pygame.display.set_caption("Reading Speed Experiment")
        self.clock = pygame.time.Clock()
        self.results: list[dict] = []
        self.trial_idx = 0
        self.state = "ready"  # "ready" | "showing_text" | "showing_question"
        self.text_start_time = 0
        self.option_rects: list[tuple[pygame.Rect, str]] = []  # (rect, answer_letter)

    def sample_trials(self) -> list[Trial]:
        return [Trial(q, w) for q, w in zip(random.sample(self.questions, self.n_trials), random.sample(range(self.wpm_range[0], self.wpm_range[1]), self.n_trials))]





    def ms_per_word_for_trial(self, wpm: int) -> float:
        """Milliseconds each word is displayed. Uses CONFIG override if set."""
        if self.config.get("ms_between_words") is not None:
            return self.config["ms_between_words"]
        return 60000.0 / wpm if wpm > 0 else 0

    def draw_centered_word(self, surf: pygame.Surface, font: pygame.font.Font, word: str):
        """Draw a single word centered on screen."""
        s = font.render(word, True, self.config["text_color"])
        x = (surf.get_width() - s.get_width()) // 2
        y = (surf.get_height() - s.get_height()) // 2
        surf.blit(s, (x, y))

    def run(self):
        """Run the experiment main loop."""
        running = True
        while running and self.trial_idx < len(self.trials):
            trial = self.trials[self.trial_idx]
            dt = self.clock.tick(60) / 1000.0

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
                        self.text_start_time = pygame.time.get_ticks() / 1000.0

                if event.type == pygame.MOUSEBUTTONDOWN and self.state == "showing_question":
                    pos = event.pos
                    for rect, letter in self.option_rects:
                        if rect.collidepoint(pos):
                            correct = letter == trial.question_item.correct_answer
                            self.results.append({
                                "trial": self.trial_idx + 1,
                                "wpm": trial.wpm,
                                "correct": correct,
                                "question": trial.question_item.question[:50],
                            })
                            self.trial_idx += 1
                            self.state = "ready"
                            self.option_rects = []
                            break

            self.screen.fill(self.config["bg_color"])

            if self.state == "ready":
                ms = self.ms_per_word_for_trial(trial.wpm)
                prompt = f"Trial {self.trial_idx + 1} of {len(self.trials)}  —  {int(ms)} ms/word"
                prompt_surf = self.font_question.render(prompt, True, self.config["accent_color"])
                self.screen.blit(
                    prompt_surf,
                    (self.screen.get_width() // 2 - prompt_surf.get_width() // 2, 80),
                )
                instruct = "Press SPACE to begin"
                inst_surf = self.font_text.render(instruct, True, self.config["text_color"])
                self.screen.blit(
                    inst_surf,
                    (self.screen.get_width() // 2 - inst_surf.get_width() // 2, 300),
                )

            elif self.state == "showing_text":
                elapsed_ms = (pygame.time.get_ticks() / 1000.0 - self.text_start_time) * 1000
                word_ms = self.ms_per_word_for_trial(trial.wpm)
                words = trial.question_item.text.split()
                total_duration_ms = len(words) * word_ms

                if elapsed_ms < total_duration_ms:
                    word_idx = int(elapsed_ms / word_ms)
                    if word_idx < len(words):
                        self.draw_centered_word(self.screen, self.font_text, words[word_idx])
                else:
                    # Brief blank before question
                    self.state = "showing_question"
                    self.option_rects = []
                    self.screen.fill(self.config["bg_color"])
                    pygame.display.flip()
                    pygame.time.delay(self.config["inter_trial_blank_ms"])

            elif self.state == "showing_question":
                q = trial.question_item
                q_surf = self.font_question.render(q.question, True, self.config["text_color"])
                self.screen.blit(q_surf, (50, 80))

                self.option_rects = []
                y = 180
                for opt in q.options:
                    letter = opt[0].upper() if opt else ""
                    opt_surf = self.font_options.render(opt, True, self.config["text_color"])
                    pad = self.config["option_padding"]
                    w = max(400, opt_surf.get_width() + pad * 2)
                    h = opt_surf.get_height() + pad * 2
                    x = (self.screen.get_width() - w) // 2
                    rect = pygame.Rect(x, y, w, h)

                    # Draw option box
                    pygame.draw.rect(self.screen, (50, 55, 60), rect, border_radius=8)
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
            ms = 60000 // r["wpm"] if r["wpm"] > 0 else 0
            print(f"Trial {r['trial']}: {r['wpm']} WPM ({ms} ms/word) — {status}")
        correct_count = sum(1 for r in self.results if r["correct"])
        print(f"\nScore: {correct_count}/{len(self.results)} correct")




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
