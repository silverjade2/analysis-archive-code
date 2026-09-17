# voice-llm-freshness

["다시 말씀해 주시겠어요", 그리고 아무도 다시 말하지 않았다](https://analysis-archive.vercel.app/analyses/voice-llm-freshness) 재현 코드.

원본은 음성 비서 운영 로그에서 LLM의 낡은 답(knowledge cutoff 이후 정보)을 발화 쪽 신호로 우회해서 세던 분석. 낡은 답은 로그에 정상 응답으로 남아서 직접 셀 수 없었음. 여기서는 낡은 답에 `truth_stale`을 붙인 가상 로그를 만들어 그 우회 신호가 실제로 몇 개를 가리키는지 맞춰 본다 (실제 로그 없음). 그림 1~3만 예외로 글에 적힌 운영 로그 기록 비율(`data/record_rates.csv`, 손으로 옮김)을 그린 것.

## 실행

```bash
pip install -r requirements.txt
for s in scripts/[01][0-9]_*.py; do python "$s"; done
python verify.py
```

10초 안쪽. 경로는 `scripts/common.py`에서 파일 위치 기준.

재현에 쓴 버전: Python 3.14.6(macOS), numpy 2.5.3, pandas 3.0.5, matplotlib 3.11.2, pillow 12.3.0. `data/`와 `outputs/`를 치우고 다시 돌려도 `data/*.csv.gz`와 `outputs/results/*.csv`는 byte 동일 (gzip mtime을 0으로 고정). png, webp는 비교 안 함.

## 스크립트

| | 역할 | 주요 출력 |
| --- | --- | --- |
| 01 | 에피소드 단위로 발화, 응답, 제어 코드, 음성 참조 행 생성. `truth_*` 4열 | `data/logs.csv.gz` |
| 02 | 유저 발화에 응답 붙이기. 제어 코드 건너뜀, 120초 컷 | `turn_pairs.csv.gz` |
| 03 | 주제 키워드 매칭, 제외 규칙, 불만 발화의 직전 주제 귀속 | `topic_match.csv.gz`, `exclude_hits.csv` |
| 04 | fallback 문구, 최신성 신호, 의도 | `turn_flags.csv.gz`, `state_by_*.csv`, `fallback_*.csv`, `overview.csv` |
| 05 | 재시도와 세션 구조, 재시도 정의 sweep | `retry_by_prev_state.csv`, `session_sweep.csv`, `session_stats.csv` |
| 06 | 세 축 곱 점수. 매칭 발화 분모와 전체 분모 두 벌 | `priority_matched.csv`, `priority_all.csv`, `coverage.csv` |
| 07 | `truth_stale` 대비 proxy recall, precision | `oracle_compare.csv`, `coverage_by_truth_topic.csv` |
| 08 | 오호출 규칙 하한과 상한, 분모를 바꾼 비율 | `misfire_bounds.csv`, `rates_by_denominator.csv` |
| 09 | 미매칭 발화 n-gram | `uncovered_ngram.csv` |
| 10 | 그림 4장. 1~3은 `record_rates.csv`, 4는 `priority_all.csv` | `outputs/figures/site/` |

사전(주제 키워드, 제외 규칙, fallback 문구, 의도 규칙)과 상수는 전부 `common.py`. 02~06, 09는 `truth_` 컬럼을 읽지 않고 07, 08만 대조에 쓴다. `verify.py`가 그 순서와 02의 짝짓기(루프로 다시 계산, 제어 코드가 끼었거나 지연 응답이 있는 세션 위주)를 확인.

`logs.csv.gz`는 `row_id, session_id, ts(0.1초), log_type, text` + `truth_misfire, truth_topic, truth_fresh, truth_stale`. log_type 0 유저, 1 기기, 2 제어 코드, 3과 4는 schema에 없는 음성 원본 참조 행. 유저 발화의 text는 빈 문자열일 수 있음. `truth_misfire`는 사람이 기기에 한 말이 아님, `truth_topic`은 키워드와 무관한 실제 주제("오늘 비 와"도 날씨), `truth_fresh`는 최신 정보가 있어야 답할 수 있는 질문, `truth_stale`은 정상 응답인데 낡은 답.

## 확인할 숫자

- 04: 빈 발화 56.2%, 원문 있는 발화의 fallback 26.9%, 무응답 7.9%
- 05: 재시도율 fallback 직후 0.57%, 정상 직후 1.43%, 무응답 직후 7.30%. fallback 직후가 가장 낮은 순서가 sweep 12개 정의 전부에서 유지
- 06: 매칭 발화 1.69%. 최신성 신호 발화의 94.7%가 미매칭. 미매칭 한 줄 점수가 1위 주제의 57배
- 07: 최신성 신호 recall 0.629 / precision 0.351. 불만 표현은 precision 0.944인데 recall 0.035, 재시도는 recall 0.016
- 08: 빈 발화를 전부 오호출로 세는 규칙의 precision 0.55

## 생성기에 넣은 구조

- 오호출은 유저 발화의 40% 안팎(생성 로그 실측 39.6%. 01의 파라미터 0.35는 에피소드 단위라 발화 단위와 다름), 그중 78%가 빈 발화이고 나머지는 대화 파편과 TV 음성. 사람 발화는 확률 0.48로 STT가 비움(이동형 기기라 거리, 주행 소음 가정. 재시도 턴은 원문을 다시 뽑아서 생성 로그 실측은 41.8%). 둘을 합쳐 운영 로그의 빈 발화 57%를 채움
- 사람도 파편을 말함. "아니", "이거 말고"는 사람이 기기에 한 말이고 의도 규칙에 안 걸림
- fallback 문구는 인식 실패 계열 14개 + 긴 꼬리 100개. 상위 두 문구 가중치는 운영 로그 비중(39.8%, 17.6%)에 맞춤, 생성 결과는 40.3%, 17.8% (`fallback_phrases.csv`). 정보 부재 문구는 정보 질문이 fallback을 받을 때만 나옴
- 낡은 답: 최신 정보가 필요한 질문이 정상 응답을 받았고, 모델 cutoff(2025-01-01, 모델 카드의 2025년 1월)부터 발화 시각까지가 주제별 유효 기간(3시간~720시간, lognormal 흔들림)을 넘으면 `truth_stale=1`. 시각과 날짜 질문은 기기 시계로 답한다고 보고 제외
- 세션은 기기가 닫음. fallback 뒤 재시도는 97%가 새 세션, 무응답 뒤는 85%가 같은 세션
- 주제 키워드에 2글자 키워드와 부분 문자열 오탐(경기도, 사회복지사, 수영하고)을 심음. 날씨 질문 대부분은 키워드 없이 들어옴("오늘 비 와")
- 따라와 명령은 2026-03-02 출시, 의도 규칙에 없음

## 알려진 문제

- 생성기 가중치는 운영 로그의 주변 분포(빈 발화, 매칭 비율, fallback 문구 쏠림, 재시도율 순서)가 나올 때까지 손으로 조정한 값. 오호출 40%와 STT 누락 확률 0.48은 그 결과이고 추정치가 아님
- cutoff가 로그 시작보다 9개월 앞이라 최신 정보가 필요한 정상 응답은 거의 다 낡은 답. 유효 기간 차이가 `truth_stale`에 거의 안 드러남 (`coverage_by_truth_topic.csv` stale_rate 0.67~0.76)
- 세션을 넘어간 재시도는 기기 식별자가 없어 못 셈. sweep은 같은 세션 안에서 창 길이와 유사도 기준만 바꿈. 무응답 직후 재시도율 7.30%는 운영 로그보다 높음 (무응답이면 기기가 계속 듣는다는 가정이 강함)
- 불만 발화가 적어 주제별 불만 precision은 몇 건으로 정해짐
