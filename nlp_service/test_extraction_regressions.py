import unittest
from unittest.mock import patch

import action_items as ai_module
import pain_points as pp_module
from action_items import ActionItemExtractor
from pain_points import PainPointExtractor


def _segments(lines: list[str]) -> list[dict]:
    segments = []
    for idx, text in enumerate(lines):
        segments.append({
            "text": text,
            "speaker": f"SPEAKER_{idx:02d}",
            "start": float(idx * 10),
        })
    return segments


class _NegativePainClassifier:
    def __call__(self, _text: str, top_k=None):
        rows = [
            {"label": "not_pain_point", "score": 0.9},
            {"label": "pain_point", "score": 0.1},
        ]
        return [rows] if top_k is None else [rows[0]]


class ActionItemRegressionTests(unittest.TestCase):
    def test_meeting_admin_lines_are_not_action_items(self):
        lines = [
            "Let's start with apologies for absence.",
            "I'll start.",
            "Oh yeah, I'm Lucy Strokes, PA to Rita.",
            "Okay, on to the next item.",
            "Well, next...",
            "Please, carry on.",
        ]
        with patch.object(ai_module, "_get_action_classifier", return_value=None), patch.object(
            ai_module, "_llm_call", return_value=None
        ):
            items = ActionItemExtractor().extract_action_items(_segments(lines))
        self.assertEqual(items, [])

    def test_real_follow_up_tasks_still_extract(self):
        lines = [
            "We need to send an email letting all the staff know there are only five spaces that belong to us.",
            "Jason, you need to park by the garages.",
            "Let's all come up with four and email them over in the next two days and I'll take it from there.",
            "I'll speak with Clive and let you know dates.",
        ]
        with patch.object(ai_module, "_get_action_classifier", return_value=None), patch.object(
            ai_module, "_llm_call", return_value=None
        ):
            items = ActionItemExtractor().extract_action_items(_segments(lines))

        self.assertEqual(len(items), 4)
        self.assertEqual(items[0]["owner"], "GROUP")
        self.assertEqual(items[1]["owner"], "Jason")
        self.assertEqual(items[1]["action"], "park by the garages")
        self.assertEqual(items[2]["owner"], "GROUP")
        self.assertEqual(items[2]["deadline"], "next two days")
        self.assertIn("Clive", items[3]["action"])

    def test_conditional_send_it_line_is_not_action_item(self):
        lines = ["If no one's listening I'll send it."]
        with patch.object(ai_module, "_get_action_classifier", return_value=None), patch.object(
            ai_module, "_llm_call", return_value=None
        ):
            items = ActionItemExtractor().extract_action_items(_segments(lines))
        self.assertEqual(items, [])


class PainPointRegressionTests(unittest.TestCase):
    def test_meeting_admin_lines_are_not_pain_points(self):
        lines = [
            "Thanks for coming to today's monthly meeting.",
            "Let's start with apologies for absence.",
            "I'll start.",
            "I'll put a sign up with my name on it and my parking space.",
        ]
        with patch.object(pp_module, "_get_classifier", return_value=None), patch.object(
            pp_module, "_llm_extract", return_value={}
        ):
            points = PainPointExtractor(threshold=0.34).extract(_segments(lines))
        self.assertEqual(points, [])

    def test_business_issues_are_detected_as_pain_points(self):
        lines = [
            "Sue has identified a growing problem with staff morale.",
            "There seems to be a significant problem with staff morale in the sales team and also sickness absence.",
            "There are issues like lack of training and lack of effective appraisals.",
        ]
        with patch.object(pp_module, "_get_classifier", return_value=None), patch.object(
            pp_module, "_llm_extract", return_value={}
        ):
            points = PainPointExtractor(threshold=0.34).extract(_segments(lines))

        self.assertEqual(len(points), 3)
        self.assertTrue(all(point["severity"] in {"high", "medium"} for point in points))

    def test_solution_and_meta_discussion_lines_are_not_pain_points(self):
        lines = [
            "Listen, if he parks by the garages in the corner where he normally parks he's normally okay there so let's have him have that parking space.",
            "Alright, so I'll put a sign up with my name on it and my parking space.",
            "Okay, so what has this got to do with staff morale?",
        ]
        with patch.object(pp_module, "_get_classifier", return_value=None), patch.object(
            pp_module, "_llm_extract", return_value={}
        ):
            points = PainPointExtractor(threshold=0.34).extract(_segments(lines))
        self.assertEqual(points, [])

    def test_negative_classifier_confidence_no_longer_counts_as_pain_signal(self):
        lines = ["Thanks for coming to today's monthly meeting."]
        with patch.object(pp_module, "_get_classifier", return_value=_NegativePainClassifier()), patch.object(
            pp_module, "_llm_extract", return_value={}
        ):
            points = PainPointExtractor(threshold=0.34).extract(_segments(lines))
        self.assertEqual(points, [])


if __name__ == "__main__":
    unittest.main()
