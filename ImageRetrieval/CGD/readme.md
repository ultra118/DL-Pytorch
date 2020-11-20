# Combination of Multiple Global Descriptors for Image Retrieval

> 보통 conv layer뒤에 FC layer를 붙여 이미지의 dimension을 줄여 Global Descriptor로 사용했음, 이런 GD에 여러가지 변형, 발전이 있음

__참조 링크:__ https://cyc1am3n.github.io/2019/05/02/combination-of-multiple-global-descriptors-for-image-retrieval.html

__paper__: https://arxiv.org/pdf/1903.10663v4.pdf

__code__: https://github.com/leftthomas/CGD

- Global pooling method: Conv layer의 activation을 활용

  - SPoC(Sum Pooling of Convolution)
  - MAC(Maximum Activation of Convolution)
  - GeM(Generalized mean Pooling)

- 성능 높이기 위한 변형

  - weighted sum pooling
  - weighted GeM
  - R-MAC

- 여러 global descriptor를 활용해, ensemble같은 효과를 내는 방식 고안

  - __CGD(Combination of multiple Global Descriptors)__
  - ![image](https://user-images.githubusercontent.com/28910538/98500387-b22ea080-228f-11eb-81aa-aed6e71f9f5b.png)
    - 첫번째는 backbone, image representation 학습
    - 두 번째는 보조 모듈로 CNN을 fine-tune하도록 도움

  

# code tip

- utils.ImageReader를 따로 define 해두고 사용하는데, 각 데이터 셋에 따라 적합하게 dataset type으로 변환해준다

  - 근데 이부분은 각 데이터 셋 특징에 맞게 조절되기때문에 custom 데이터 셋으로 그냥 편하게 쓰기위해 ImageFolder로 dataset 생성하기로 함

  - 이렇게 함에따라 따로 각 input image에 대한 각 image/label을 변수로 정의해서 사용하는 부분이 있어 이는 아래 처럼 따로 정의해두고 사용함

    - ```python
      test_data_set = ImageFolder(f'{data_path}/val', set_transforms(data_type='test', input_resolution=112))
      
      test_data_set_labels = list(map(lambda x: x[1], list(iter(test_data_set))))
      test_data_set_images = list(map(lambda x: x[0], list(iter(test_data_set))))
      ```


- model은 fc layer 이전까지,  avg pool 빼고 layer 4~

- resnet18 model의 layer 비교해보니

  - CMD에 추가되어 있는 parameter들은

    - `total_params`, `total_ops`가 각 layer마다 추가되어 있음
    - model 구조 파악 필요해 보임

  - __신기한 건 그냥 종단 layer쯤에 이런 gd(glboal descriptor)들이 추가된 거일텐데 파라미터 뽑아보면 layer 중간 중간에 param 값들이 있다는 거임__

    - ```python
      def forward(self, x):
          # backbone, layer4~ avg pool 이전 까지
          shared = self.features(x)
          global_descriptors = []
          """
          self.global_descriptors
          ModuleList(
            (0): GlobalDescriptor(p=1) -> S 
            
            if self.p == 1:
            	return x.mean(dim=[-1, -2])
            	(1,512,7,7)로 들어와서 (1,512)로 나옴
            (1): GlobalDescriptor(p=3) -> G
            else:
            	sum_value = x.pow(self.p).mean(dim=[-1, -2])
            	return torch.sign(sum_value) * (torch.abs(sum_value).pow(1.0 / self.p))
          )
          self.auxiliary_module
              Sequential(
            (0): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): Linear(in_features=512, out_features=21, bias=True)
          )
      	self.main_modules
          ModuleList(
            (0): Sequential(
              (0): Linear(in_features=512, out_features=128, bias=False)
              (1): L2Norm()
            )
            (1): Sequential(
              (0): Linear(in_features=512, out_features=128, bias=False)
              (1): L2Norm()
            )
          )
          """
          for i in range(len(self.global_descriptors)):
              global_descriptor = self.global_descriptors[i](shared)
              if i == 0:
                  classes = self.auxiliary_module(global_descriptor)
                  global_descriptor = self.main_modules[i](global_descriptor)
                  global_descriptors.append(global_descriptor)
                  global_descriptors = F.normalize(torch.cat(global_descriptors, dim=-1), dim=-1)
                  return global_descriptors, classes
      ```

    - nn.modulelist는 list로 module받아서 layer 구성하는 것, sequential과 다른 점은 forward가 선언되어 있지 않음, nn.Sequential은 list로는 못받음, 직접 다 indexing 해줘야함

- resnet18로 model 학습했을때 가정


  - ```python
    # CGD model
    model = Model(backbone_type, gd_config, feature_dim, num_classes=len(train_data_set.class_to_idx)).cuda()
    torch.save(model.state_dict(), '{}/{}_{}_epoch_model.pth'.format(opt.log_save_dir, save_name_pre, epoch))
    ```

  - 학습 코드는 분명 그냥 model.state_dict()부분을 저장하는데,
    나중에 학습된 weights 써먹으려고보면 len(keys())가 맞지않음
    `len(model.state_dict().keys())=128`,
    `len(torch.load(weights).keys())=181`


    - 근데 model.state_dict()에 descriptor weight는 없는 것 같음, 그리고 `.total_ops`, `_total_params`같은 값들이 저장되어 있음

      - profile 함수 통해  ops, parameter 저장됨, ops는 operator system?
    - `model.load_state_dict(strict=False)` , strict인자에 false를 주면 load 하는 state_dict과 model의 state_dict중 중복되는 경우의 layer만 load함

  - 일반적인 resnet18은 122개임, CGD는 fc layer부분이 없고 auxiliary_modules로 projection

    

- model(intput) 값을 넣어 inference를 하는 경우 batch size를 1보다 크게 넣을때는 inference하지만, 1의 값을 넣을때 오류가 발생하는 경우가 있음

  - https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
  - 이 경우 모델 어딘가에서 배치의 평균이나 표준 값을 통한 계산이 필요한 경우, `nn.BatchNorm`같은 경우에 따라 에러가 발생하는 경우임, 이럴떈 `model.eval`호출함으로써,  현재 배치에대해 계산하는 대신 실행중인 추정치를 사용해 레이어 동작을 변경함