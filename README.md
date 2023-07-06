# AI CONNECT: Fake or Real AI 생성 이미지 판별 경진대회

<img alt="Html" src ="https://img.shields.io/badge/AICONNECT Final rank-Top 11/0.8960-yellow?style=for-the-badge"/>

#### 생성 AI가 만들어낸 가짜 (Fake) 이미지와 진짜 (Real) 이미지를 분류하는 문제 (23.05.24  - 23.07.20) - 김준용, 길다영
##### 📊 [PUBLIC] 11/115 점수: 0.8966
##### 📊 [PRIVATE] 11/115 점수: 0.8957

<br><br>

### File 설명

- <b>CODE_README.md</b> 참고

<br>

### EfficientNet_b0 사용
- Baseline에서 제안된 efficientnet_b0을 그대로 사용. <br>
  - efficientnet 버전을 높여 사용해 봤으나, 오히려 성능이 떨어짐.


<br>

### 성능 향상 방법
#### 1. Optimizer 교환 (train_config.yaml)
- adam → adamw

#### 2. val dataset size 줄임 (train_config.yaml)
- 0.3 → 0.2

#### 3. Data augmentation 사용 (datasets.py)
- train dataset에 다양한 data augmentation 기법 적용.

```
self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456,0.406],[0.229,0.224,0.225])
        ])
```

#### 4. epoch 늘림
- 20 → 250

#### 5. 외부 데이터 사용
- Kaggle | [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) 추가

<br>


### 아쉬운 점
- 시간 부족으로 외부 데이터를 추가한 Datasets과 다음 data augmentation을 추가한 코드를 epoch 250으로 돌려보지 못함.

  ```
  transforms.ColorJitter(brightness=(0.5, 0.9),
                                     contrast=(0.4, 0.8),
                                     saturation=(0.7, 0.9),
                                     hue=(-0.2, 0.2),
                                     ),
  ```

- 대회가 끝난 후 [커뮤니티](https://aiconnect.kr/competition/detail/227/task/295/community/detail/185)에서 AI CONNECT 연습문제였던 "소고기 이미지를 통한 등급 분류 문제" 의 baseline을 참고하여 classifier을 수정한다면 더 좋은 성능이 나온다는 것을 확인.

  ```
  class EffNet(nn.Module):
      def __init__(self, n_outputs:int, **kwargs):
          super(EffNet, self).__init__()
          self.model = timm.create_model('efficientnet_b4', pretrained=True)
          self.model.classifier = nn.Sequential(
              nn.Linear(in_features = 1792, out_features=625),
              nn.ReLU(),
              nn.Dropout(p=0.3),
              nn.Linear(in_features=625, out_features=256),
              nn.ReLU(),
              nn.Linear(in_features=256, out_features=n_outputs)
          )
          
      def forward(self, x):
          output = self.model(x)
          return output
  ```


<br><br>


<b>출처 |</b> [AI CONNECT - Fake or Real: AI 생성 이미지 판별 경진대회](https://aiconnect.kr/competition/detail/227) <br>
