# session-wrap-usage-log

[무엇을 자동화할 것인가: 쓰면서 느낀 필요를 대화 기록으로 확인한 세션 마감 도구](https://analysis-archive.vercel.app/analyses/session-wrap-usage-log) 재현 코드.

Claude Code 대화 기록 한 달치에서 사람이 입력한 메시지만 걸러 의도별로 세고, 그 집계로 만들 도구를 고른 분석의 재현. 실제 대화 기록은 없음. 기록 형식(user 레코드의 tool_result, isMeta, 서브에이전트 파일, 슬래시 명령, IDE 자동 삽입, 다른 세션이 쓴 첫 지시문)과 프로젝트 구성만 옮겨 가상 기록을 만들고, 거르기, 정규식 군집, 후보 비교를 같은 순서로 돌린 뒤 생성기가 기록한 의도와 대조한다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
for s in scripts/0*.py; do .venv/bin/python "$s"; done
```

10초 안쪽. Python 3.14.7, numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2, pillow 12.3.0. 이 조합에서 `data/`와 `outputs/`를 지우고 다시 돌려도 생성 데이터, 결과 CSV, 그림이 byte 단위로 같았음.

## 스크립트

| | 역할 | 주요 출력 |
|---|---|---|
| 01 | 프로젝트 10개, 30일, 본 세션 229개와 서브에이전트 기록 생성. 문장은 `phrases.txt` | `data/transcripts.jsonl.gz`, `truth_parts.csv` |
| 02 | 사람 메시지 거르기 (1차 접두어, 2차 IDE 자동 삽입과 다른 세션이 쓴 첫 지시문), 슬래시 명령과 스킬 호출 세기 | `data/messages.csv`, `funnel.csv`, `project_messages.csv`, `slash_commands.csv`, `skill_calls.csv` |
| 03 | 정규식 군집. 겹침 허용, 첫 일치, 규칙 순서 200가지의 첫 일치 범위 | `data/labels.csv`, `cluster_counts.csv`, `count_summary.csv`, `cluster_by_project.csv` |
| 04 | 기타 군집의 짧은 응답에 직전 assistant 텍스트나 앞 메시지의 군집을 붙임, /clear 직전 메시지 | `data/labels_ctx.csv`, `clear_before.csv`, `other_breakdown.csv`, `context_counts.csv`, `clear_context.csv` |
| 05 | 후보 4개가 덮는 메시지 수 (겹침 합, 합집합, 첫 일치, 짧은 응답 포함), 퍼진 프로젝트 수, 스킬 호출 대비 대화 건수 | `candidates.csv`, `skill_vs_dialog.csv` |
| 06 | ground truth 평가. 거르기, 군집별 precision/recall, 기타의 실제 구성, 후보별 실제 크기 | `eval_*.csv` |
| 07 | 그림 5장 | `outputs/figures/site/fig1~5` |

`truth_parts.csv`는 06만 읽음. 02~05는 기록 텍스트와 메타 필드만 봄.

판단 기준은 `common.py`에: 규칙 15개와 첫 일치 순서(세션 관리 규칙이 커밋보다 앞), 짧은 응답은 공백 뺀 10자 이하, 후보가 퍼졌다고 볼 프로젝트당 메시지 3건.

## 확인할 숫자

- 02: user 내용 조각 18,226 -> 1차 통과 2,529 -> 사람 메시지 2,386. tool_result 14,039, IDE 자동 삽입 131
- 03: 커밋/푸시/배포 겹침 336, 첫 일치 280 (순서 200가지에서 246~336). Claude Code 사용법 첫 일치 2차 전 269, 뒤 138
- 04: 기타 895 중 짧은 응답 742. 붙인 뒤 커밋/푸시/배포 280 -> 456
- 05: A(커밋/푸시/배포 + 세션 마감) 겹침 합 500, 합집합 463, 첫 일치 434, 짧은 응답 포함 648, 프로젝트 9곳
- 06: 첫 일치 정확도 0.620, 붙인 뒤 0.715. A 실제 755. C 실제 80, D 실제 123 (정규식은 C 94, D 73)

## 알려진 문제

- 의도 정답은 생성기가 정한 것. 정규식이 놓치거나 잘못 잡는 비율은 `phrases.txt`의 문장 구성에 따라 바뀜
- 확인 질문 뒤 짧은 응답 확률은 커밋/푸시에 가장 높게 넣었음 (0.4, 나머지 0.12~0.3). 실제 기록에서 센 값이 아님
