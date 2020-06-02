import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.utils.mem import *

from torchvision.models import vgg16_bn

path = untar_data(URLs.PETS) #데이터 다운로드
path_hr = path/'images' #원본 이미지 경로
path_lr = path/'small-96' #96*96 이미지 경로
path_mr = path/'small-256' #25*256 이미지가 경로

il = ImageList.from_folder(path_hr) #이미지리스트 만들기

def resize_one(fn, i, path, size): #이미지 크기 조절
    dest = path/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True) #비율에 맞게 이미지를 최소 사이즈로 바꿈
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    img.save(dest, quality=60)

# create smaller image sets the first time this nb is run
sets = [(path_lr, 96), (path_mr, 256)]
for p,size in sets:
    if not p.exists():
        print(f"resizing to {size} into {p}")
        parallel(partial(resize_one, path=p, size=size), il.items)


bs,size=32,128 #batch size: 32 / image size: 128
arch = models.resnet34 #resnet34 모델 사용

src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42) #96*96 이미지의 10퍼센트를 검증 데이터로.

def get_data(bs,size):
    data = (src.label_from_func(lambda x: path_hr/x.name) #labeling
           .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True) #zoom 2배
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True)) #normalize

    data.c = 3
    return data

data = get_data(bs,size)


data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9,9))   #데이터 예시

#Feature loss

t = data.valid_ds[0][1].data #검증 데이터 셋 데이터를 가져오기
t = torch.stack([t,t]) #배치사이즈 2개짜리 배치 가상적으로 만들기


def gram_matrix(x):
    n,c,h,w = x.size() #n: 데이터 개수, c: 채널, h: 세로, w: 가로 사이즈 가져오기
    x = x.view(n, c, -1) #reshape하기 x.view(n,c,h*w)
    return (x @ x.transpose(1,2))/(c*h*w) #전치행렬: 행과 열을 교환한 행렬

gram_matrix(t)

base_loss = F.l1_loss #손실함수

vgg_m = vgg16_bn(True).features.cuda().eval() #vgg16 사용해 features값 convolution
requires_grad(vgg_m, False) #grad 사용하지 않음

blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)] #이 코드는 어떤 기능을 하나요?
blocks, [vgg_m[i] for i in blocks]


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target): # input: 저화질  target: 고화질
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)] # 각 픽셀의 loss를 계산
        self.feat_losses += [base_loss(f_in, f_out) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()

feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2]) #vgg model 사용

#Train

wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect();

learn.lr_find()
learn.recorder.plot() #그래프

lr = 1e-3

def do_fit(save_name, lrs=slice(lr), pct_start=0.9): #학습->저장->결과 보여주기
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)

do_fit('1a', slice(lr*10))

learn.unfreeze()
do_fit('1b', slice(1e-5,lr))

data = get_data(12,size*2) #이미지 사이즈 2배로

learn.data = data
learn.freeze()
gc.collect()

learn.load('1b');
do_fit('2a')

learn.unfreeze()
do_fit('2b', slice(1e-6,1e-4), pct_start=0.3) #prediction이 더 선명해짐

#Test

learn = None
gc.collect();

free = gpu_mem_get_free_no_cache()
# the max size of the test image depends on the available GPU RAM
if free > 8000: size=(1280, 1600) # >  8GB RAM
else:           size=( 820, 1024) # <= 8GB RAM
print(f"using size={size}, have {free}MB of GPU RAM free")

learn = unet_learner(data, arch, loss_func=F.l1_loss, blur=True, norm_type=NormType.Weight)

data_mr = (ImageImageList.from_folder(path_mr).split_by_rand_pct(0.1, seed=42)
          .label_from_func(lambda x: path_hr/x.name)
          .transform(get_transforms(), size=size, tfm_y=True)
          .databunch(bs=1).normalize(imagenet_stats, do_y=True))
data_mr.c = 3

learn.load('2b');
learn.data = data_mr

fn = data_mr.valid_ds.x.items[0]; fn #test해볼 이미지

img = open_image(fn); img.shape
p,img_hr,b = learn.predict(img)

show_image(img, figsize=(18,15), interpolation='nearest');    #저해상도 이미지 체크

Image(img_hr).show(figsize=(18,15))  #고해상도 이미지로 변환됨