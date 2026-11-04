"""Exp 08-D: direct evidence selection prompt ablation."""

from LLM_test import exp08d_selector_probe as probe

# Preserve original enum schema, candidate data,
# model and generation parameters.
# Change only the selector's instructions.

probe.SYSTEM += (
    "근거를 선택할 때 단순한 주제 유사성과 "
    "질문에 대한 직접적인 설명을 구분하세요. "
    "질문이 판단 기준을 물으면 판단 조건을, "
    "방법을 물으면 구체적인 방법을, "
    "시점을 물으면 해당 시점을 설명하는 "
    "구절을 우선 선택하세요. "
    "질문과 관련된 회계 용어가 등장하더라도 "
    "서로 다른 회계 판단이나 측정 대상을 "
    "설명하는 문장은 선택하지 마세요. "
    "직접 근거가 없으면 NONE을 선택하세요."
)

if __name__ == "__main__":
    probe.main()
