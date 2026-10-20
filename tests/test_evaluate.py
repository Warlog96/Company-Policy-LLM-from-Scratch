"""Tests for the evaluation metric helpers."""
from eval.evaluate import answer_correct, key_values, is_refusal


def test_key_values():
    assert key_values("The lunch break is 1 hour, taken between 12:30 PM and 2:30 PM.") == ["1", "12:30", "2:30"]
    assert key_values("Contact helpdesk@arcvale.com, INR 1,500.") == ["helpdesk@arcvale.com", "1500"]
    assert key_values("From the 21st of one month to the 20th.") == ["21st", "20th"]


def test_answer_correct_whole_values():
    ref = "Employees get 12 days of casual leave per year."
    assert answer_correct("As per company policy, employees get 12 days of casual leave.", ref)
    assert not answer_correct("Employees get 2 days.", "Employees get 12 days")
    assert not answer_correct("Employees get 12 days.", "Employees get 2 days")


def test_answer_correct_ordinals():
    ref = "The attendance cycle runs from the 21st of one month to the 20th of the next month."
    assert answer_correct("The attendance cycle runs from the 21st of one month to the 20th of the next month.", ref)
    assert not answer_correct("It runs from the 1st to the 30th.", ref)


def test_answer_correct_without_numbers():
    ref = "Alcohol is not reimbursed under any expense category."
    assert answer_correct("As per company policy, alcohol is not reimbursed under any expense category.", ref)
    assert not answer_correct("Passwords must be changed regularly.", ref)


def test_is_refusal():
    assert is_refusal("I don't know. This is not covered in the company policies.")
    assert not is_refusal("Employees get 12 days of casual leave per year.")
