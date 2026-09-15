# voice-llm-freshness

["다시 말씀해 주시겠어요", 그리고 아무도 다시 말하지 않았다](https://analysis-archive.vercel.app/analyses/voice-llm-freshness) 재현 코드.

knowledge cutoff가 있는 LLM이 낡은 답을 하면 로그에는 정상 응답으로 남는다. 운영 로그에서는 그 실패를 발화 쪽 신호로만 우회해서 셌다. 여기서는 낡은 답에 `truth_stale`을 붙인 가상 음성 로그를 만들어, 그 우회 신호가 낡은 답을 실제로 몇 개나 가리키는지 맞춰 본다. 실제 로그는 없다. 그림 1, 2, 3만 예외로, 글에 적힌 운영 로그 기록 비율을 `data/record_rates.csv`에 옮겨 그렸다.

## 실행

```bash
pip install -r requirements.txt
for s in scripts/[01][0-9]_*.py; do python "$s"; done
python verify.py
```

10초 안쪽. 경로는 `scripts/common.py`에서 파일 위치 기준으로 잡는다.

환경: Python 3.14.6(macOS), numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.2, pillow 12.3.0. `data/`와 `outputs/`를 치우고 다시 돌려도 `data/*.csv.gz`와 `outputs/results/*.csv`가 byte 단위로 같았다. png, webp는 비교하지 않았다.

## pipeline

| | 하는 일 | 출력 |
|---|---|---|
| 01 | 에피소드 단위로 발화, 응답, 제어 코드, 음성 참조 행 생성 | `data/logs.csv.gz` |
| 02 | 유저 발화에 응답 붙이기. 제어 코드 건너뜀, 120초 컷 | `data/turn_pairs.csv.gz` |
| 03 | 주제 키워드 매칭, 제외 규칙, 불만 발화의 직전 주제 귀속 | `data/topic_match.csv.gz`, `exclude_hits.csv` |
| 04 | fallback 문구, 최신성 신호, 의도 | `data/turn_flags.csv.gz`, `state_by_*.csv`, `fallback_*.csv`, `overview.csv` |
| 05 | 재시도와 세션 구조, 재시도 정의 sweep | `retry_by_prev_state.csv`, `session_sweep.csv`, `session_stats.csv` |
| 06 | 세 축 곱 점수. 매칭 발화 분모와 전체 분모 | `priority_matched.csv`, `priority_all.csv`, `coverage.csv` |
| 07 | `truth_stale` 대비 proxy recall, precision | `oracle_compare.csv`, `coverage_by_truth_topic.csv` |
| 08 | 오호출 규칙 하한과 상한, 분모를 바꾼 비율 | `misfire_bounds.csv`, `rates_by_denominator.csv` |
| 09 | 미매칭 발화 n-gram | `uncovered_ngram.csv` |
| 10 | 그림 4장 (`sitestyle.py`). 1~3은 `data/record_rates.csv`, 4는 `priority_all.csv` | `outputs/figures/site/` |

`verify.py`는 02~06, 09에 `truth_` 문자열이 없는지, 중간 테이블에 정답 컬럼이 없는지, 08이 규칙을 다 만든 뒤에 정답을 읽는지, 02의 짝짓기가 행을 하나씩 도는 루프와 같은지를 본다. 짝짓기는 제어 코드가 끼었거나 지연 응답이 있는 세션을 골라 대조한다.

## logs.csv.gz

| 컬럼 | |
|---|---|
| `row_id`, `session_id`, `ts` | ts는 0.1초 단위 |
| `log_type` | 0 유저 발화, 1 기기 발화, 2 제어 코드. 3, 4는 schema에 없는 음성 원본 참조 행 |
| `text` | 유저 발화는 빈 문자열일 수 있다 |
| `truth_misfire` | 사람이 기기에 한 말이 아님 |
| `truth_topic` | 키워드와 무관한 실제 주제. "오늘 비 와"도 날씨 |
| `truth_fresh` | 최신 정보가 있어야 맞게 답할 수 있는 질문 |
| `truth_stale` | 정상 응답을 받았지만 그 답이 낡음 |

`truth_*`는 07, 08의 대조에만 쓴다.

`data/record_rates.csv`는 생성 데이터가 아니다. 운영 로그 분석 기록의 비율을 손으로 옮긴 파일이고 그림 1~3만 읽는다.

## 가상데이터에 넣은 구조

- 오호출은 유저 발화의 40% 안팎. 오호출의 78%는 빈 발화이고 나머지는 대화 파편과 TV 음성
- 사람 발화의 48%는 STT가 비운 빈 발화. 이동형 기기라 거리와 주행 소음으로 인식이 끊긴다고 가정했다. 운영 로그의 빈 발화 57%는 오호출과 이쪽을 합쳐 채운다
- 사람도 파편을 말한다. 답을 끊는 "아니", "아니야", "이거 말고"는 사람이 기기에 한 말이고 의도 규칙에는 걸리지 않는다
- fallback 문구는 인식 실패 계열 14개와 긴 꼬리 100개. 상위 두 문구 가중치는 운영 로그 비중에 맞췄다. 정보 부재 문구는 정보 질문이 fallback을 받을 때만 나온다
- 낡은 답: 최신 정보가 필요한 질문이 정상 응답을 받으면, 모델 cutoff(2025-01-01, 모델 카드의 2025년 1월)부터 발화 시각까지 걸린 시간이 주제별 유효 기간(3시간에서 720시간, lognormal 흔들림)을 넘을 때 `truth_stale=1`. 시각과 날짜 질문은 기기 시계로 답한다고 보고 뺐다
- 세션은 기기가 닫는다. fallback 뒤 재시도는 97%가 새 세션으로 넘어가고, 무응답 뒤에는 85%가 같은 세션에 남는다
- 주제 키워드에 2글자 키워드와 부분 문자열 오탐(경기도, 사회복지사, 수영하고)을 넣었다. 날씨 질문 대부분은 키워드 없이 들어온다("오늘 비 와", "우산 챙겨야 돼")
- 따라와 명령은 2026-03-02 출시. 의도 규칙에는 없다

## 확인할 숫자

- 04: 빈 발화 56.2%, 원문 있는 발화의 fallback 26.9%, 무응답 7.9%
- 05: 재시도율 fallback 직후 0.57%, 정상 직후 1.43%, 무응답 직후 7.30%. fallback 직후가 가장 낮아야 생성기가 운영 로그만큼 지저분하다는 뜻이고, sweep 12개 정의 전부에서 그 순서가 유지된다
- 06: 매칭 발화 1.69%. 최신성 신호 발화의 94.7%가 미매칭
- 06: 미매칭 한 줄 점수가 1위 주제의 57배. 최신성 요구율은 미매칭이 가장 낮은데도 그렇다
- 07: 최신성 신호 recall 0.629, precision 0.351. 불만 표현은 precision 0.944인데 recall 0.035, 재시도는 recall 0.016으로 거의 못 잡는다
- 08: 빈 발화를 전부 오호출로 세는 규칙의 precision 0.55. 빈 발화의 절반 가까이가 사람 쪽이라서다

## 결과 파일

| 파일 | 내용 |
|---|---|
| `overview.csv` | 빈 발화, 응답 상태, 최신성 신호 구성 |
| `state_by_fresh.csv`, `state_by_intent.csv` | 최신성, 의도별 응답 상태 |
| `fallback_phrases.csv`, `fallback_categories.csv` | 문구 순위와 계열 비중 |
| `exclude_hits.csv` | 제외 규칙별 취소 건수 |
| `retry_by_prev_state.csv`, `session_sweep.csv`, `session_stats.csv` | 재시도와 세션 |
| `priority_matched.csv`, `priority_all.csv`, `coverage.csv` | 점수표 두 벌과 매칭 범위 |
| `oracle_compare.csv`, `coverage_by_truth_topic.csv` | 정답 대비 proxy, 실제 주제별 키워드 coverage |
| `misfire_bounds.csv`, `rates_by_denominator.csv` | 오호출 규칙과 분모별 비율 |
| `uncovered_ngram.csv` | 미매칭 발화 상위 30 |

## 남은 것

- 생성기 가중치는 운영 로그의 주변 분포(빈 발화, 매칭 비율, fallback 문구 쏠림, 재시도율 순서)가 나올 때까지 손으로 조정했다. 오호출 40%와 사람 쪽 빈 발화 48%는 조정의 결과이지 추정한 비율이 아니다
- cutoff가 로그 시작보다 9개월 앞이라 최신 정보가 필요한 정상 응답은 거의 다 낡은 답이 된다. 유효 기간 차이는 `truth_stale`에 거의 안 드러난다(`coverage_by_truth_topic.csv`의 stale_rate 0.67~0.76)
- 세션을 넘어간 재시도는 기기 식별자가 없어 셀 수 없다. sweep은 같은 세션 안에서 창 길이와 유사도 기준만 바꾼다
- 무응답 직후 재시도율(7.30%)은 운영 로그보다 높다. 무응답이면 기기가 계속 듣는다는 가정이 강하게 들어갔다
- 불만 발화는 수가 적어서 주제별 불만 precision은 몇 건으로 정해진다
