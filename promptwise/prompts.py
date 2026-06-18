"""Module for handling prompt templates and candidates.

Defines the Prompt dataclass and builds a pool of candidate prompts
for a doctor-appointment assistant.  Token costs are computed with
tiktoken's cl100k_base encoding (the same BPE vocabulary used by
Claude-class models).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import tiktoken

# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the exact token count of *text* using cl100k_base."""
    return len(_ENCODING.encode(text))


# ---------------------------------------------------------------------------
# Prompt dataclass
# ---------------------------------------------------------------------------

PromptCategory = Literal["instruction", "example", "domain_context"]


@dataclass
class Prompt:
    """A single candidate prompt that may be included in an LLM call.

    Attributes
    ----------
    id : str
        Unique human-readable identifier (e.g. ``"instr_tone"``).
    category : PromptCategory
        One of ``"instruction"``, ``"example"``, or ``"domain_context"``.
    text : str
        The prompt content that will be injected into the LLM context.
    tags : list[str]
        Latent query types this prompt is designed to help with (e.g.
        ``"booking"``, ``"availability"``).
    token_cost : int
        Number of tokens in *text*, computed automatically via tiktoken.
    """

    id: str
    category: PromptCategory
    text: str
    tags: list[str]
    token_cost: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_cost = count_tokens(self.text)


# ---------------------------------------------------------------------------
# Prompt pool
# ---------------------------------------------------------------------------

_PROMPT_DEFS: list[dict] = [
    # ── Instruction prompts (~15-50 tokens) ─────────────────────────────
    {
        "id": "instr_tone",
        "category": "instruction",
        "text": (
            "You are a polite, professional medical receptionist. "
            "Always use a warm and reassuring tone."
        ),
        "tags": ["general_info", "booking", "availability"],
    },
    {
        "id": "instr_confirm_booking",
        "category": "instruction",
        "text": (
            "Before finalising any appointment, repeat the date, time, "
            "doctor name, and reason for visit back to the patient and "
            "ask for explicit confirmation."
        ),
        "tags": ["booking"],
    },
    {
        "id": "instr_escalate_emergency",
        "category": "instruction",
        "text": (
            "If the patient describes chest pain, difficulty breathing, "
            "severe bleeding, or any life-threatening symptom, immediately "
            "instruct them to call emergency services (911) and do NOT "
            "attempt to book a routine appointment."
        ),
        "tags": ["emergency"],
    },
    {
        "id": "instr_privacy",
        "category": "instruction",
        "text": (
            "Never share a patient's medical history, insurance details, "
            "or personal information with anyone other than the patient "
            "themselves. Comply with HIPAA at all times."
        ),
        "tags": ["general_info", "insurance"],
    },
    {
        "id": "instr_verify_identity",
        "category": "instruction",
        "text": (
            "Verify the patient's full name and date of birth before "
            "accessing or modifying any appointment records."
        ),
        "tags": ["booking", "cancellation", "general_info"],
    },
    {
        "id": "instr_clarify_ambiguity",
        "category": "instruction",
        "text": (
            "If the patient's request is ambiguous, ask a clarifying "
            "question rather than guessing. Never assume a preferred "
            "doctor or time slot."
        ),
        "tags": ["booking", "availability", "general_info"],
    },
    {
        "id": "instr_no_medical_advice",
        "category": "instruction",
        "text": (
            "Do not provide medical diagnoses or treatment advice. "
            "If the patient asks a clinical question, recommend they "
            "discuss it with their doctor during the appointment."
        ),
        "tags": ["general_info", "emergency"],
    },

    # ── Example-based few-shot prompts (~100-150 tokens) ────────────────
    {
        "id": "example_booking_dialogue",
        "category": "example",
        "text": (
            "Example — Booking a new appointment:\n"
            "Patient: I'd like to see a dermatologist sometime next week.\n"
            "Assistant: Sure! Dr. Priya Sharma (Dermatology) has openings "
            "on Tuesday at 10:00 AM and Thursday at 2:30 PM. Which works "
            "better for you?\n"
            "Patient: Tuesday morning, please.\n"
            "Assistant: Great — I have you down for Tuesday, March 18 at "
            "10:00 AM with Dr. Priya Sharma for a dermatology consultation. "
            "Can you confirm this is correct?\n"
            "Patient: Yes, confirmed.\n"
            "Assistant: Your appointment is booked. You'll receive a "
            "confirmation text shortly."
        ),
        "tags": ["booking", "availability"],
    },
    {
        "id": "example_reschedule_dialogue",
        "category": "example",
        "text": (
            "Example — Rescheduling an existing appointment:\n"
            "Patient: I need to move my Thursday appointment with "
            "Dr. Chen to next week.\n"
            "Assistant: Of course. Dr. Chen has availability on Monday "
            "at 9:00 AM and Wednesday at 3:00 PM next week. Would "
            "either of those work?\n"
            "Patient: Monday at 9 works.\n"
            "Assistant: Done — your appointment has been rescheduled "
            "to Monday, March 24 at 9:00 AM with Dr. Chen. "
            "I've cancelled your original Thursday slot. "
            "Please note our 24-hour cancellation policy still applies."
        ),
        "tags": ["booking", "cancellation"],
    },
    {
        "id": "example_cancellation_dialogue",
        "category": "example",
        "text": (
            "Example — Cancelling an appointment:\n"
            "Patient: I want to cancel my appointment tomorrow with "
            "Dr. Okafor.\n"
            "Assistant: I can help with that. Since your appointment is "
            "within 24 hours, please note that a $25 late-cancellation "
            "fee may apply per our clinic policy. Would you still like "
            "to proceed with the cancellation?\n"
            "Patient: Yes, go ahead.\n"
            "Assistant: Your appointment with Dr. Okafor on March 15 at "
            "11:00 AM has been cancelled. If you'd like to rebook, "
            "just let me know."
        ),
        "tags": ["cancellation"],
    },

    # ── Domain-context prompts (~30-80 tokens) ──────────────────────────
    {
        "id": "ctx_doctor_specialties",
        "category": "domain_context",
        "text": (
            "Available doctors and specialties:\n"
            "• Dr. Priya Sharma — Dermatology (Mon, Tue, Fri)\n"
            "• Dr. James Chen — Internal Medicine (Mon–Thu)\n"
            "• Dr. Amara Okafor — Pediatrics (Tue, Wed, Fri)\n"
            "• Dr. Robert Kim — Orthopedics (Mon, Wed, Thu)\n"
            "• Dr. Lisa Fernandez — Cardiology (Tue, Thu)"
        ),
        "tags": ["availability", "booking", "general_info"],
    },
    {
        "id": "ctx_doctor_schedules",
        "category": "domain_context",
        "text": (
            "Doctor schedule details:\n"
            "Morning slots: 9:00 AM, 9:30 AM, 10:00 AM, 10:30 AM, "
            "11:00 AM, 11:30 AM.\n"
            "Afternoon slots: 1:00 PM, 1:30 PM, 2:00 PM, 2:30 PM, "
            "3:00 PM, 3:30 PM, 4:00 PM.\n"
            "Each appointment is 30 minutes. Last booking at 4:00 PM."
        ),
        "tags": ["availability", "booking"],
    },
    {
        "id": "ctx_insurance_accepted",
        "category": "domain_context",
        "text": (
            "Insurance plans accepted: BlueCross BlueShield, Aetna, "
            "Cigna, UnitedHealthcare, Medicare, and Medicaid. "
            "Patients with other plans should contact billing at "
            "(555) 012-3456 to verify coverage before booking."
        ),
        "tags": ["insurance", "general_info"],
    },
    {
        "id": "ctx_clinic_hours",
        "category": "domain_context",
        "text": (
            "Sunrise Medical Clinic\n"
            "Address: 742 Evergreen Terrace, Suite 200, Springfield, IL 62704\n"
            "Hours: Monday–Friday 8:30 AM – 5:00 PM, Saturday 9:00 AM – 1:00 PM, "
            "closed Sunday.\n"
            "Phone: (555) 867-5309."
        ),
        "tags": ["general_info"],
    },
    {
        "id": "ctx_cancellation_policy",
        "category": "domain_context",
        "text": (
            "Cancellation policy: Appointments cancelled more than 24 hours "
            "in advance incur no fee. Cancellations within 24 hours are "
            "subject to a $25 late-cancellation fee. No-shows are charged "
            "a $50 fee. Three consecutive no-shows may result in account "
            "suspension."
        ),
        "tags": ["cancellation"],
    },
    {
        "id": "ctx_appointment_types",
        "category": "domain_context",
        "text": (
            "Appointment types offered:\n"
            "• New patient visit (60 min)\n"
            "• Follow-up visit (30 min)\n"
            "• Urgent same-day visit (30 min, limited availability)\n"
            "• Annual physical / wellness check (45 min)\n"
            "• Telehealth video consultation (30 min)"
        ),
        "tags": ["booking", "availability", "general_info"],
    },
    {
        "id": "ctx_new_vs_returning",
        "category": "domain_context",
        "text": (
            "New patients must complete registration forms and provide "
            "a valid photo ID and insurance card at least 15 minutes "
            "before their first appointment. Returning patients should "
            "check in 5 minutes early and report any insurance changes."
        ),
        "tags": ["booking", "insurance", "general_info"],
    },
    {
        "id": "ctx_referral_requirements",
        "category": "domain_context",
        "text": (
            "Referrals: Cardiology and Orthopedics appointments require "
            "a referral from a primary care physician. Dermatology, "
            "Pediatrics, and Internal Medicine accept self-referrals."
        ),
        "tags": ["booking", "general_info"],
    },
    {
        "id": "ctx_followup_rules",
        "category": "domain_context",
        "text": (
            "Follow-up scheduling: After each visit the doctor may "
            "recommend a follow-up. Follow-ups should be booked before "
            "leaving the clinic when possible. Telehealth follow-ups "
            "are available for stable patients."
        ),
        "tags": ["booking", "general_info"],
    },
    {
        "id": "ctx_urgent_care_guidelines",
        "category": "domain_context",
        "text": (
            "For non-emergency urgent issues (fever, minor injuries, "
            "rashes), same-day urgent slots may be available. Call the "
            "clinic by 9:30 AM to request an urgent slot. If none are "
            "open, the nearest urgent-care centre is Springfield "
            "Urgent Care at 100 Main Street, (555) 999-0101."
        ),
        "tags": ["emergency", "availability", "general_info"],
    },
]


def load_prompt_pool() -> list[Prompt]:
    """Instantiate and return the full pool of candidate prompts.

    Token costs are computed automatically via tiktoken on construction.
    """
    return [Prompt(**defn) for defn in _PROMPT_DEFS]
