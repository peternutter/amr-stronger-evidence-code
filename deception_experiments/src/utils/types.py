"""Some common type defintions"""

Message = dict[str, str]

Conversation = list[Message]


class Label:
    HONEST = 0
    DECEPTIVE = 1

    @classmethod
    def from_str(cls, label_str: str) -> int:
        s = label_str.lower().strip()
        if s == "honest" or s == str(cls.HONEST) or s.startswith("hon"):
            return cls.HONEST
        elif s == "deceptive" or s == str(cls.DECEPTIVE) or s.startswith("decept"):
            return cls.DECEPTIVE
        else:
            raise ValueError(f"Invalid label: {label_str}")

    @classmethod
    def to_str(cls, label: int) -> str:
        if label == cls.HONEST:
            return "honest"
        elif label == cls.DECEPTIVE:
            return "deceptive"
        else:
            raise ValueError(f"Invalid label: {label}")
