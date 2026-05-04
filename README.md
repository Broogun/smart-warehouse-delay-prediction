# 스마트 창고 출고 지연 예측 AI

DACON [스마트 창고 출고 지연 예측 AI 경진대회](https://dacon.io) 풀이입니다.

- **기간**: 2026.04.01 ~ 2026.05.04
- **참가자**: 1,006명
- **평가 지표**: MAE (Mean Absolute Error)
- **Public MAE**: 9.98526 — 608팀 중 **81위** (상위 13.3%)
- **Private MAE**: 10.21288 — 607팀 중 **82위** (상위 13.5%)

## 문제 정의

스마트 창고의 센서 데이터 및 운영 지표를 기반으로 **향후 30분간 평균 출고 지연 시간(분)** 을 예측합니다.

- 하나의 시나리오 = 25개의 타임슬롯 (15분 단위 스냅샷)
- 학습/예측 데이터 모두 시나리오 내 25개 슬롯이 동시에 제공됨

## 모델 구조

```
[ Level-1 ]
  LightGBM (15-Fold GroupKFold)  → OOF predictions
  CatBoost (15-Fold GroupKFold)  → OOF predictions

[ Level-2 ]
  LightGBM Meta-model
  Input: Level-1 OOF preds + Top-100 original features
  → Final prediction
```

## 핵심 기법

| 기법 | 설명 |
|------|------|
| log1p transform | 타깃에 log1p 적용 후 학습, 예측 시 expm1 역변환 |
| GroupKFold | scenario_id 단위 분할로 데이터 누수 방지 |
| Scenario Aggregate | 시나리오 전체 25슬롯 집계 피처 (mean/max/std) |
| Lead Features | 미래 슬롯 값 활용 (25슬롯 동시제공이므로 누수 아님) |
| Sample Weighting | 고지연(≥40분) 샘플 가중치 강화 + 시나리오 단위 multiplier |
| LGBM + CatBoost Stacking | 두 모델의 OOF 예측값을 메타 모델 입력으로 활용 |

## 실험 결과

| 실험 | Public MAE | 비고 |
|------|-----------|------|
| LGBM + CatBoost Stacking (GPU 튜닝) | **9.9852** | 최종 제출 |
| 미사용 컬럼 추가 (staff_on_floor 등) | 10.0125 | 소폭 악화 |
| CPU Optuna 재튜닝 | 10.0153 | GPU 파라미터보다 나쁨 |
| Layout target encoding | 10.3747 | test 분포 불일치로 악화 |
| Bias correction | 10.7335 | OOF bias ≠ test bias |
| Pseudo-labeling | 10.9163 | 노이즈 레이블로 악화 |

## 에러 분석 결과

OOF 잔차 분석을 통해 발견한 모델 약점:

- **고지연(≥40분) bias: -33.7** → 극단 시나리오 심각한 과소예측
- **후반 슬롯(19-24) MAE**: 초반(0-5) 대비 약 1.5배
- 시간 관계상 분석 후 재학습은 미진행 (sample weight 강화 코드는 적용된 상태)

## 파일 구조

```
├── configs/            # 모델 하이퍼파라미터 (yaml)
│   ├── lgbm.yaml
│   ├── catboost.yaml
│   └── meta.yaml
├── src/                # 핵심 모듈
│   ├── config.py       # 경로 및 상수 정의
│   ├── features.py     # 피처 엔지니어링
│   ├── train.py        # LightGBM 학습
│   ├── train_catboost.py
│   └── predict.py      # 예측
├── notebooks/
│   ├── 00_problem_definition.ipynb  # 문제 정의 및 컬럼 정의
│   └── 01_eda.ipynb                 # EDA 및 피처 엔지니어링 분석
├── run_train.py        # LightGBM 학습 실행
├── run_catboost.py     # CatBoost 학습 실행
├── run_stacking.py     # 스태킹 메타 모델 학습
├── run_predict.py      # 예측 및 제출 파일 생성
├── run_error_analysis.py  # OOF 잔차 분석
└── run_optuna.py       # 하이퍼파라미터 튜닝
```

## 실행 순서

```bash
python run_train.py        # LightGBM 학습 → OOF 예측값 생성
python run_catboost.py     # CatBoost 학습 → OOF 예측값 생성
python run_stacking.py     # 메타 모델 학습 + 최종 예측 및 제출 파일 생성
```
