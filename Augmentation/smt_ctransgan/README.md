# SMT cTransGAN

AIHub SMT의 diffusion 실험과 같은 분할, 정상 부품 자리 선택, JPEG 저장 규칙을 사용한다.
공정별로 해당 K장만 사용해 cTransGAN을 새로 학습한다. core.py는 실행 당시 VisA cTransGAN 학습 코드를 복사하고 클래스 수를 공정별로 바꾼 버전이다.

학과 서버의 26_2_dl 환경에서 저장소 루트 기준으로 실행한다.

```bash
cd Augmentation
python -m smt_ctransgan.generate --k 1 --seed 0
cd ..
python -m AnomalyDetection.smt.seg_ad --category pre --k 1 --seeds 0 1 2 --conds none ctransgan:1:0.5 ctransgan:5:0.5 ctransgan:10:0.5 ctransgan:20:0.5 ctransgan:40:0.5
```

K={1,4,16}, seed={0,1,2}, 공정별 종류당 40장으로 총 6,480쌍을 생성했다.
사전 6종·납땜 12종이며 조건당 720쌍이다. 생성 후 공용 누수 검사를 적용했다.
납땜공정 평가에는 --category solder를 사용한다.
생성 이미지, 원본, 체크포인트는 AIHub 파생물 규칙에 따라 서버의 /home/junho 아래에 보관한다.

결과는 AnomalyDetection/results/smt/ctransgan_comparison.csv의 108행과 ctransgan_summary.md를 참고한다.
평가 지표는 과검률 1%에서의 미검률이며 3개 시드 평균을 비교한다.
사전공정과 납땜 K1·K4에서는 개선된 조건이 있으나 납땜 K16은 모든 생성 장수에서 악화됐다.
일부 K16 학습에서 판별기 손실이 0에 가까워지는 불안정성이 관찰됐다.
이미지는 원본 결함과 유사하게 재구성될 수 있어, 결과를 새로운 결함 형태의 생성 능력으로 단정하지 않는다.
