# AIHub SMT 원본 데이터 (git 미포함)

이 폴더에는 AIHub **「SMT 공정 개선 멀티모달 데이터」(데이터셋 번호 71900)** 원본이 들어간다.

## 왜 저장소에 없는가

AIHub 이용약관상 데이터를 **제3자에게 제공(재배포)할 수 없다.** 그래서 원본(`raw/`)은 물론,
원본으로 만든 생성 불량 이미지(`Dataset/aihub_smt/generated/`)와 SMT 사진이 들어간 발표 그림
(`ppt_figures/local_only/`)도 파생물로 보고 git과 공유 Drive에서 모두 뺐다(`.gitignore`).

아래 목록·분할표도 원본 파일 이름과 라벨을 한 장씩 그대로 담고 있어 **공개 저장소에는 넣지 않았다.**
팀 서버에만 있다. 저장소에 들어 있는 SMT 관련 파일은 실험 코드(`AnomalyDetection/smt/`, `Augmentation/smt_*`)와
결과 표(`AnomalyDetection/results/smt/*.csv`, 지표만 담음)뿐이다.

| 파일 (서버에만) | 내용 |
|---|---|
| `Dataset/aihub_smt/index.csv` | 원본 이미지 전체 목록(경로, 라벨, 불량 종류, 촬영 코드, 이미지 통계, SHA-256) |
| `Dataset/aihub_smt/splits/*.csv` | 실험 분할표(test, train_normal, val_normal, extra_normal, pool, train_K{1,2,4,8,16,32}_seed{0..4}) |
| `Dataset/aihub_smt/splits/splits_meta.json` | 분할 설정(seed 7017)과 종류별 개수 |
| `Dataset/aihub_smt/check_shortcut.csv` | 촬영 조건만으로 정상/불량이 구분되는지 확인한 결과 |

SMT 실험을 다시 돌리려면 AIHub에서 원본을 직접 받고(아래), 위 분할표는 팀에 따로 요청해야 한다.
분할표의 경로가 그대로 맞도록 원본은 아래처럼 푼다.

## 받는 방법

1. https://www.aihub.or.kr 에 가입하고 「SMT 공정 개선 멀티모달 데이터」(71900) 페이지에서 **활용 신청**을 한다. 승인이 나야 받을 수 있다.
2. AIHub 명령줄 도구 `aihubshell`을 받는다(우리는 v0.6, 25.09.19 버전을 썼다).
   ```bash
   curl -o aihubshell https://api.aihub.or.kr/api/aihubshell.do && chmod +x aihubshell
   ./aihubshell -help                                   # 옵션 확인
   ./aihubshell -mode l -datasetkey 71900               # 파일 목록과 filekey 확인
   ./aihubshell -mode d -datasetkey 71900 -filekey <필요한 파일 키, 쉼표로 구분> -aihubapikey '<본인 API 키>'
   ```
   웹 페이지에서 직접 받아도 된다.
3. **필요한 파일만 받으면 된다.** 실험은 이미지와 라벨만 쓴다.
   - 필요: `Training/01.원천데이터/TS_이미지데이터_*`, `Training/02.라벨링데이터/TL_*`,
     `Validation/01.원천데이터/VS_이미지데이터_*`, `Validation/02.라벨링데이터/VL_*`
     (사전공정·납땜공정, 정상·불량 전부)
   - 불필요: `TS_센서데이터_*`, `VS_센서데이터_*` (xlsx, 실험에 쓰지 않음)
4. 분할 받은 파일(`*.zip.part0`, `*.zip.part1` …)이 남아 있으면 합친다: `cat X.zip.part* > X.zip`
5. **zip마다 zip 이름과 같은 폴더를 만들어 그 안에 푼다.** 분할표 경로가 이 폴더 이름을 그대로 쓴다.
   ```bash
   cd Dataset/aihub_smt/raw/18.SMT_공정_개선_멀티모달_데이터/3.개방데이터/1.데이터
   for z in */*/*.zip; do d="${z%.zip}"; mkdir -p "$d" && python -m zipfile -e "$z" "$d" && rm "$z"; done
   ```

## 풀고 난 뒤 모양

```
Dataset/aihub_smt/raw/18.SMT_공정_개선_멀티모달_데이터/3.개방데이터/1.데이터/
├─ Training/
│  ├─ 01.원천데이터/TS_이미지데이터_{사전공정,납땜공정}_{정상,불량}_{번호}.{종류}/*.jpg|JPG
│  └─ 02.라벨링데이터/TL_{사전공정,납땜공정}_{정상,불량}_{번호}.{종류}/*.json
└─ Validation/
   ├─ 01.원천데이터/VS_이미지데이터_.../
   └─ 02.라벨링데이터/VL_.../
```

예: `Training/01.원천데이터/TS_이미지데이터_사전공정_불량_4.납볼/PR_DEF_PB_A_20250902-122135_00903.jpg`

이미지+라벨은 약 6.7GB(9만 2천여 개 파일)다. 이 중 분할표가 실제로 쓰는 것은 21,110개(약 1.8GB)이고,
나머지는 `index.csv`를 다시 만들거나 분할을 다시 뽑을 때만 필요하다.

## 확인

분할표(팀에서 받은 것)가 가리키는 이미지·라벨이 모두 있고 SHA-256이 맞는지 본다. `missing 0, bad 0`이 나오면 된다.

```bash
python - <<'EOF'
import csv, glob, hashlib, os
miss = bad = n = 0
for f in glob.glob('Dataset/aihub_smt/splits/*.csv'):
    for r in csv.DictReader(open(f, encoding='utf-8')):
        for k in ('image', 'label_json'):
            if r.get(k) and not os.path.exists(r[k]): miss += 1
        if os.path.exists(r['image']):
            n += 1
            if hashlib.sha256(open(r['image'], 'rb').read()).hexdigest() != r['image_sha256']: bad += 1
print('rows', n, 'missing', miss, 'bad', bad)
EOF
```

## 생성 이미지(`Dataset/aihub_smt/generated/`)

생성 불량도 위 이유로 저장소에 없다. 원본을 받은 뒤 `Augmentation/smt_ctransgan/`, `Augmentation/smt_sd_lora/`의
스크립트로 다시 만든다(출력 경로 `Dataset/aihub_smt/generated/{방법}/K{k}_seed{s}/{images,masks,manifest.csv}`).
