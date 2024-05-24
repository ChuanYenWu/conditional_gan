"# conditional_gan" 
spider_crop
stackgan_stage1, stage2, stagewhole
stackgan_util

## Conditional GAN(Generative Adversarial Network)
GAN模型能夠透過noise產生圖片，此project將控制部分輸入GAN的變因，透過輸入文字給予生成圖片一些限制(conditional GAN)，達成簡單的text-to-image。
<br>
採用動漫人物人臉(Anime Face)，輸入的文字條件包含髮色、瞳色(hair, eyes)

### 資料準備
spider_crop.ipynb
<br>
* 資料抓取: 透過圖庫網站提供的API。進行網路爬蟲，抓取json內容，選出符合條件的圖片(如:髮色、瞳色tag只有一組)，讀取圖片
* 臉部切割: [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface)所提供的動漫臉部偵測xml來擷取出圖片中的臉部部分(由於此xml是2011年作成，針對後來演變的畫風、光影或上色方式，可能會有些擷取錯誤的產生，此project沒有進行人工剔除)
* 取得256x256圖片約3萬8千張，抓取的tag範圍有12種髮色和11種瞳色，其分布可見下方圖表(more than single表示該圖有超過1種的髮色或瞳色tag，此種圖不在抓取範圍內，因此數量為0；no information表示該圖缺少髮色或瞳色tag，只要有其中1類tag就會抓取)
![_tag_distribute]()
