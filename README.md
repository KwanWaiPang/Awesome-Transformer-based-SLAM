<p align="center">
  <h1 align="center">
  Awesome Transformer-based SLAM
  </h1>
</p>

This repository contains a curated list of resources addressing SLAM related task employing Transformer, including optical flow, view/feature correspondences, stereo matching, depth estimation, 3D reconstruction, pose estimation, etc.

If you find some ignored papers, **feel free to [*create pull requests*](https://github.com/KwanWaiPang/Awesome-Transformer-based-SLAM/blob/pdf/How-to-PR.md), or [*open issues*](https://github.com/KwanWaiPang/Awesome-Transformer-based-SLAM/issues/new)**. 

Contributions in any form to make this list more comprehensive are welcome.

If you find this repositorie is useful, a simple star ([![Github stars](https://img.shields.io/github/stars/KwanWaiPang/Awesome-Transformer-based-SLAM.svg)](https://github.com/KwanWaiPang/Awesome-Transformer-based-SLAM)) should be the best affirmation. 😊

Feel free to share this list with others!

# Overview

- [Transformer-based SLAM](#Transformer-based-SLAM)
- [Transformer-based Pose Tracking](#Transformer-based-Pose-Tracking)
- [Transformer-based Optical Flow](#Transformer-based-Optical-Flow)
- [Transformer-based View Matching](#Transformer-based-View-Matching)
- [Transformer-based Mapping](#Transformer-based-Mapping)


## Transformer-based SLAM

Full SLAM, including pose and depth

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`CVPR`|[SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos](https://arxiv.org/pdf/2412.09401)|[![Github stars](https://img.shields.io/github/stars/PKU-VCL-3DV/SLAM3R.svg)](https://github.com/PKU-VCL-3DV/SLAM3R)|[test](https://kwanwaipang.github.io/SLAM3R/)|
|2025|`CVPR`|[MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](https://arxiv.org/pdf/2412.12392)|[![Github stars](https://img.shields.io/github/stars/rmurai0610/MASt3R-SLAM.svg)](https://github.com/rmurai0610/MASt3R-SLAM)|[Website](https://edexheim.github.io/mast3r-slam/) <br> [Test](https://kwanwaipang.github.io/MASt3R-SLAM/)
|2022|`ECCV`|[Jperceiver: Joint perception network for depth, pose and layout estimation in driving scenes](https://arxiv.org/pdf/2207.07895)|[![Github stars](https://img.shields.io/github/stars/sunnyHelen/JPerceiver.svg)](https://github.com/sunnyHelen/JPerceiver)|---|


## Transformer-based Pose Tracking

or pose/state estimation
<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`arXiv`|[XIRVIO: Critic-guided Iterative Refinement for Visual-Inertial Odometry with Explainable Adaptive Weighting](https://arxiv.org/pdf/2503.00315)|---|---|
|2025|`IEEE Acess`|[Transformer-based model for monocular visual odometry: a video understanding approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10845764)|[![Github stars](https://img.shields.io/github/stars/aofrancani/TSformer-VO.svg)](https://github.com/aofrancani/TSformer-VO)|---|
|2025|`arXiv`|[Light3R-SfM: Towards Feed-forward Structure-from-Motion](https://arxiv.org/pdf/2501.14914)|---|---|
|2024|`arXiv`|[MASt3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion](https://arxiv.org/pdf/2409.19152)|[![Github stars](https://img.shields.io/github/stars/naver/mast3r.svg)](https://github.com/naver/mast3r/tree/mast3r_sfm)|MASt3R sfm version|
|2024|`CVPR`|[VGGSfM: Visual Geometry Grounded Deep Structure From Motion](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_VGGSfM_Visual_Geometry_Grounded_Deep_Structure_From_Motion_CVPR_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/vggsfm.svg)](https://github.com/facebookresearch/vggsfm)|[website](https://vggsfm.github.io/)|
|2024|Msc Thesis|[End-to-End Learned Visual Odometry Based on Vision Transformer](https://www.utupub.fi/bitstream/handle/10024/178848/Aman_Manishbhai_Vyas_Thesis.pdf?sequence=1)|---|---|
|2024|`arXiv`|[Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry](https://arxiv.org/pdf/2409.08769)|[![Github stars](https://img.shields.io/github/stars/ybkurt/VIFT.svg)](https://github.com/ybkurt/VIFT)|---|
|2024|`IJRA`|[DDETR-SLAM: A Transformer-Based Approach to Pose Optimization in Dynamic Environments](https://assets-eu.researchsquare.com/files/rs-2965479/v1_covered_409a1161-fe39-4b94-9411-68639c8215b1.pdf)|---|---|
|2023|`ITM Web of Conferences`|[ViT VO-A Visual Odometry technique Using CNN-Transformer Hybrid Architecture](https://www.itm-conferences.org/articles/itmconf/pdf/2023/04/itmconf_I3cs2023_01004.pdf)|---|---|
|2023|`arXiv`|[TransFusionOdom: interpretable transformer-based LiDAR-inertial fusion odometry estimation](https://arxiv.org/pdf/2304.07728)|[![Github stars](https://img.shields.io/github/stars/RakugenSon/Multi-modal-dataset-for-odometry-estimation.svg)](https://github.com/RakugenSon/Multi-modal-dataset-for-odometry-estimation)|---|
|2023|`CVPR`|[Modality-invariant Visual Odometry for Embodied Vision](https://openaccess.thecvf.com/content/CVPR2023/papers/Memmel_Modality-Invariant_Visual_Odometry_for_Embodied_Vision_CVPR_2023_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/memmelma/VO-Transformer.svg)](https://github.com/memmelma/VO-Transformer)|[Website](https://memmelma.github.io/vot/)|
|2023|`MAV`|[ViTVO: Vision Transformer based Visual Odometry with Attention Supervision](https://elsalab.ai/publications/2023/ViTVO_Vision_Transformer_based_Visual_Odometry_with_Attention_Supervision.pdf)|---|---|
|2023|`International Conference on Haptics and Virtual Reality`|[VIOFormer: Advancing Monocular Visual-Inertial Odometry Through Transformer-Based Fusion](https://github.com/KwanWaiPang/Awesome-Transformer-based-SLAM/blob/pdf/file/978-3-031-56521-2.pdf)|---|---|
|2022|`IEEE Intelligent Vehicles Symposium`|[Attention guided unsupervised learning of monocular visual-inertial odometry](https://github.com/KwanWaiPang/Awesome-Transformer-based-SLAM/blob/pdf/file/Attention_Guided_Unsupervised_learning_of_Monocular_Visual-inertial_Odometry.pdf)|---|---|
|2022|`IEEE-SJ`|[Ema-vio: Deep visual–inertial odometry with external memory attention](https://arxiv.org/pdf/2209.08490)|---|---|
|2022|`IROS`|[AFT-VO: Asynchronous fusion transformers for multi-view visual odometry estimation](https://arxiv.org/pdf/2206.12946)|---|---|
|2022|`arXiv`|[Dense prediction transformer for scale estimation in monocular visual odometry](https://arxiv.org/pdf/2210.01723)|---|---|
|2021|`Neural Computing and Applications`|[Transformer guided geometry model for flow-based unsupervised visual odometry](https://arxiv.org/pdf/2101.02143)|---|---|

## Transformer-based Optical Flow 

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2024|`ECCV`|[Cotracker: It is better to track together](https://arxiv.org/pdf/2307.07635)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/co-tracker.svg)](https://github.com/facebookresearch/co-tracker)|---|
|2023|`arXiv`|[Win-win: Training high-resolution vision transformers from two windows](https://arxiv.org/pdf/2310.00632)|---|---|
|2023|`arXiv`|[Flowformer: A transformer architecture and its masked cost volume autoencoding for optical flow](https://arxiv.org/pdf/2306.05442)|---|---|
|2023|`CVPR`|[FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2023/papers/Shi_FlowFormer_Masked_Cost_Volume_Autoencoding_for_Pretraining_Optical_Flow_Estimation_CVPR_2023_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/XiaoyuShi97/FlowFormerPlusPlus.svg)](https://github.com/XiaoyuShi97/FlowFormerPlusPlus)|---|
|2023|`CVPR`|[Transflow: Transformer as flow learner](http://openaccess.thecvf.com/content/CVPR2023/papers/Lu_TransFlow_Transformer_As_Flow_Learner_CVPR_2023_paper.pdf)|---|---|
|2023|`ICCV`|[Croco v2: Improved cross-view completion pre-training for stereo matching and optical flow](https://openaccess.thecvf.com/content/ICCV2023/papers/Weinzaepfel_CroCo_v2_Improved_Cross-view_Completion_Pre-training_for_Stereo_Matching_and_ICCV_2023_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/naver/croco.svg)](https://github.com/naver/croco)|Cross view Match|
|2022|`NIPS`|[Croco: Self-supervised pre-training for 3d vision tasks by cross-view completion](https://proceedings.neurips.cc/paper_files/paper/2022/file/16e71d1a24b98a02c17b1be1f634f979-Paper-Conference.pdf)|[![Github stars](https://img.shields.io/github/stars/naver/croco.svg)](https://github.com/naver/croco)|Cross view Match|
|2023|`PAMI`|[Unifying flow, stereo and depth estimation](https://arxiv.org/pdf/2211.05783)|[![Github stars](https://img.shields.io/github/stars/autonomousvision/unimatch.svg)](https://github.com/autonomousvision/unimatch)|---|
|2022|`CVPR`|[Gmflow: Learning optical flow via global matching](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GMFlow_Learning_Optical_Flow_via_Global_Matching_CVPR_2022_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/haofeixu/gmflow.svg)](https://github.com/haofeixu/gmflow)|---|
|2022|`CVPR`|[Craft: Cross-attentional flow transformer for robust optical flow](http://openaccess.thecvf.com/content/CVPR2022/papers/Sui_CRAFT_Cross-Attentional_Flow_Transformer_for_Robust_Optical_Flow_CVPR_2022_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/askerlee/craft.svg)](https://github.com/askerlee/craft)|---|
|2022|`CVPR`|[Learning optical flow with kernel patch attention](https://openaccess.thecvf.com/content/CVPR2022/papers/Luo_Learning_Optical_Flow_With_Kernel_Patch_Attention_CVPR_2022_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/megvii-research/KPAFlow.svg)](https://github.com/megvii-research/KPAFlow)|---|
|2022|`CVPR`|[Global Matching with Overlapping Attention for Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Global_Matching_With_Overlapping_Attention_for_Optical_Flow_Estimation_CVPR_2022_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/xiaofeng94/GMFlowNet.svg)](https://github.com/xiaofeng94/GMFlowNet)|---|
|2022|`CVPR`|[Flowformer: A transformer architecture for optical flow](https://arxiv.org/pdf/2203.16194)|[![Github stars](https://img.shields.io/github/stars/drinkingcoder/FlowFormer-Official.svg)](https://github.com/drinkingcoder/FlowFormer-Official)|---|



## Transformer-based View Matching

or Data Assoication, Correspondences

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`arXiv`|[Loop Closure from Two Views: Revisiting PGO for Scalable Trajectory Estimation through Monocular Priors](https://arxiv.org/pdf/2503.16275)|---| MASt3R for Loop Closure|
|2025|`arXiv`|[Speedy MASt3R](https://arxiv.org/pdf/2503.10017)|---|---|
|2025|`CVPR`|[VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/pdf/2503.11651)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/vggt.svg)](https://github.com/facebookresearch/vggt)|[website](https://vgg-t.github.io/)<br>[Test](https://kwanwaipang.github.io/VGGT/)|
|2024|`ECCV`|[Grounding Image Matching in 3D with MASt3R](https://arxiv.org/pdf/2406.09756)|[![Github stars](https://img.shields.io/github/stars/naver/mast3r.svg)](https://github.com/naver/mast3r)| [Website](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/) <br> [Test](https://kwanwaipang.github.io/File/Blogs/Poster/MASt3R-SLAM.html)
|2024|`CVPR`|[RoMa: Robust dense feature matching](https://openaccess.thecvf.com/content/CVPR2024/papers/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/Parskatt/RoMa.svg)](https://github.com/Parskatt/RoMa)|---|
|2022|`ICARM`|[Tlcd: A transformer based loop closure detection for robotic visual slam](https://howardli0816.github.io/files/TLCD_A_Transformer_based_Loop_Closure_Detection_for_Robotic_Visual_SLAM.pdf)|---|---|
|2021|`ICCV`|[Cotr: Correspondence transformer for matching across images](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_COTR_Correspondence_Transformer_for_Matching_Across_Images_ICCV_2021_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/ubc-vision/COTR.svg)](https://github.com/ubc-vision/COTR)|---| 
|2021|`CVPR`|[LoFTR: Detector-free local feature matching with transformers](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/zju3dv/LoFTR.svg)](https://github.com/zju3dv/LoFTR)|---|
|2020|`CVPR`|[Superglue: Learning feature matching with graph neural networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/magicleap/SuperGluePretrainedNetwork.svg)](https://github.com/magicleap/SuperGluePretrainedNetwork)|borrows the self-attention|


## Transformer-based Mapping

or depth estimation or 3D reconstruction

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2025|`CVPR`|[Sonata: Self-Supervised Learning of Reliable Point Representations](https://arxiv.org/pdf/2503.16429)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/sonata.svg)](https://github.com/facebookresearch/sonata)|[website](https://xywu.me/sonata/)|
|2024|`CVPR`|[Point transformer v3: Simpler faster stronger](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/Pointcept/PointTransformerV3.svg)](https://github.com/Pointcept/PointTransformerV3)|---|
|2022|`NIPS`|[Point transformer v2: Grouped vector attention and partition-based pooling](https://proceedings.neurips.cc/paper_files/paper/2022/file/d78ece6613953f46501b958b7bb4582f-Paper-Conference.pdf)|[![Github stars](https://img.shields.io/github/stars/Pointcept/PointTransformerV2.svg)](https://github.com/Pointcept/PointTransformerV2) |---| 
|2021|`ICCV`|[Point transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)|---|[unofficial implementation](https://github.com/POSTECH-CVLab/point-transformer)|
|2025|`arXiv`|[Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction](https://arxiv.org/pdf/2503.16318)|---|[website](https://www.robots.ox.ac.uk/%CB%9Cvgg/research/dynamic-pointmaps/) <br> Dynamic DUSt3R, DPM|
|2025|`ICLR`|[MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion](https://arxiv.org/pdf/2410.03825)|[![Github stars](https://img.shields.io/github/stars/Junyi42/monst3r.svg)](https://github.com/Junyi42/monst3r)|[website](https://monst3r-project.github.io/)<br>[Test](https://kwanwaipang.github.io/MonST3R/)|
|2025|`CVPR`|[Stereo4D: Learning How Things Move in 3D from Internet Stereo Videos](https://arxiv.org/pdf/2412.09621)|[![Github stars](https://img.shields.io/github/stars/Stereo4d/stereo4d-code.svg)](https://github.com/Stereo4d/stereo4d-code)|[website](https://stereo4d.github.io/)|
|2025|`CVPR`|[Continuous 3D Perception Model with Persistent State](https://arxiv.org/pdf/2501.12387?)|[![Github stars](https://img.shields.io/github/stars/CUT3R/CUT3R.svg)](https://github.com/CUT3R/CUT3R)|[website](https://cut3r.github.io/)<br>CUT3R|
|2025|`CVPR`|[SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](https://arxiv.org/pdf/2411.12592)|[![Github stars](https://img.shields.io/github/stars/snldmt/SPARS3R.svg)](https://github.com/snldmt/SPARS3R)|MASt3R+COLMAP+3DGS|
|2025|`arXiv`|[SplatVoxel: History-Aware Novel View Streaming without Temporal Training](https://arxiv.org/pdf/2503.14698)|---|---|
|2025|`CVPR`|[GaussTR: Foundation Model-Aligned Gaussian Transformer for Self-Supervised 3D Spatial Understanding](https://arxiv.org/pdf/2412.13193)|[![Github stars](https://img.shields.io/github/stars/hustvl/GaussTR.svg)](https://github.com/hustvl/GaussTR)|3DGS+Transformer|
|2025|`CVPR`|[DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers](https://arxiv.org/pdf/2503.14405)|[![Github stars](https://img.shields.io/github/stars/naver/dune.svg)](https://github.com/naver/dune)|[website](https://europe.naverlabs.com/research/publications/dune/)<br>distillation|
|2025|`arXiv`|[MUSt3R: Multi-view Network for Stereo 3D Reconstruction](https://arxiv.org/pdf/2503.01661)|[![Github stars](https://img.shields.io/github/stars/naver/must3r.svg)](https://github.com/naver/must3r)|multiple views DUSt3R|
|2025|`CVPR`|[Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass](https://arxiv.org/pdf/2501.13928)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/fast3r.svg)](https://github.com/facebookresearch/fast3r)| [Website](https://fast3r-3d.github.io/) <br> [Test](https://kwanwaipang.github.io/Fast3R/)
|2024|`CVPR`|[Depth anything: Unleashing the power of large-scale unlabeled data](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Depth_Anything_Unleashing_the_Power_of_Large-Scale_Unlabeled_Data_CVPR_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/LiheYoung/Depth-Anything.svg)](https://github.com/LiheYoung/Depth-Anything)|[Website](https://depth-anything.github.io/)
|2024|`CVPR`|[DeCoTR: Enhancing Depth Completion with 2D and 3D Attentions](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_DeCoTR_Enhancing_Depth_Completion_with_2D_and_3D_Attentions_CVPR_2024_paper.pdf)|---|---|
|2024|`CVPR`|[Learning to adapt clip for few-shot monocular depth estimation](https://openaccess.thecvf.com/content/WACV2024/papers/Hu_Learning_To_Adapt_CLIP_for_Few-Shot_Monocular_Depth_Estimation_WACV_2024_paper.pdf)|---|---| 
|2024|`arXiv`|[3d reconstruction with spatial memory](https://arxiv.org/pdf/2408.16061)|[![Github stars](https://img.shields.io/github/stars/HengyiWang/spann3r.svg)](https://github.com/HengyiWang/spann3r)|[website](https://hengyiwang.github.io/projects/spanner)<br>Spann3R|
|2024|`CVPR`|[DUSt3R: Geometric 3D Vision Made Easy](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/naver/dust3r.svg)](https://github.com/naver/dust3r)| [Website](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/) <br> [Test](https://kwanwaipang.github.io/File/Blogs/Poster/MASt3R-SLAM.html)
|2024|`ECCV`|[Gs-lrm: Large reconstruction model for 3d gaussian splatting](https://arxiv.org/pdf/2404.19702)|---|[website](https://sai-bi.github.io/project/gs-lrm/)<br>3DGS+Transformer|
|2024|`TIP`|[BinsFormer: Revisiting Adaptive Bins for Monocular Depth Estimation](https://arxiv.org/pdf/2204.00987)|[![Github stars](https://img.shields.io/github/stars/zhyever/Monocular-Depth-Estimation-Toolbox.svg)](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox)|---|
|2024|`TIP`|[GLPanoDepth: Global-to-Local Panoramic Depth Estimation](https://arxiv.org/pdf/2202.02796)|---|---|
|2023|`ICCV`|[Towards zero-shot scale-aware monocular depth estimation](https://openaccess.thecvf.com/content/ICCV2023/papers/Guizilini_Towards_Zero-Shot_Scale-Aware_Monocular_Depth_Estimation_ICCV_2023_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/tri-ml/vidar.svg)](https://github.com/tri-ml/vidar)|[website](https://sites.google.com/view/tri-zerodepth)|
|2023|`ICCV`|[Egformer: Equirectangular geometry-biased transformer for 360 depth estimation](https://openaccess.thecvf.com/content/ICCV2023/papers/Yun_EGformer_Equirectangular_Geometry-biased_Transformer_for_360_Depth_Estimation_ICCV_2023_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/noahzn/Lite-Mono.svg)](https://github.com/noahzn/Lite-Mono)|---|
|2023|`Machine Intelligence Research`|[Depthformer: Exploiting long-range correlation and local information for accurate monocular depth estimation](https://link.springer.com/content/pdf/10.1007/s11633-023-1458-0.pdf)|---|---|
|2023|`CVPR`|[Lite-mono: A lightweight cnn and transformer architecture for self-supervised monocular depth estimation](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Lite-Mono_A_Lightweight_CNN_and_Transformer_Architecture_for_Self-Supervised_Monocular_CVPR_2023_paper.pdf)|---|---|
|2023|`CVPR`|[CompletionFormer: Depth Completion with Convolutions and Vision Transformers](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_CompletionFormer_Depth_Completion_With_Convolutions_and_Vision_Transformers_CVPR_2023_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/youmi-zym/CompletionFormer.svg)](https://github.com/youmi-zym/CompletionFormer)|[website](https://youmi-zym.github.io/projects/CompletionFormer/)|
|2023|`ICRA`|[Lightweight monocular depth estimation via token-sharing transformer](https://arxiv.org/pdf/2306.05682)|---|---|
|2023|`AAAI`|[ROIFormer: Semantic-Aware Region of Interest Transformer for Efficient Self-Supervised Monocular Depth Estimation](https://d1wqtxts1xzle7.cloudfront.net/104480606/25173-libre.pdf?1690190871=&response-content-disposition=inline%3B+filename%3DROIFormer_Semantic_Aware_Region_of_Inter.pdf&Expires=1741856732&Signature=KUxZHd6ZmNPg8XrNv3m~pPy~vdm9zxVdSFmbVrrYb~ZO0XTVpbbkHMgYNa05AQHpHA6NE7YckuF85Oa~rNBfT3LoMWiPm~UIxIk5zzFj6jevZsEe7WY33hUOfYeW~4JbRdYhpBN1U1zAyM4APilqFNRQMrinJ6CYmdrgoHaW6Afb5Xr2jNknzZ6zbVkB4ot26OreDLphqzyyHnmdH2YsOzbd2hTimakiibYNsY97axBqpY-u54BWhJW7-b8vtSC250M19hInvXTD79oHySYw7IUuCXwVJ4~UkJK~8ZTKVDtt3gSwMlqrKkZVv7pdzyLgbCpOHS~1VtA26sWmzyf4Hg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)|---|---|
|2023|`ICRA`|[TODE-Trans: Transparent Object Depth Estimation with Transformer](https://arxiv.org/pdf/2209.08455)|[![Github stars](https://img.shields.io/github/stars/yuchendoudou/TODE.svg)](https://github.com/yuchendoudou/TODE)|---|
|2023|`AAAI`|[Deep digging into the generalization of self-supervised monocular depth estimation](https://arxiv.org/pdf/2205.11083)|[![Github stars](https://img.shields.io/github/stars/sjg02122/MonoFormer.svg)](https://github.com/sjg02122/MonoFormer)|---|
|2022|`ECCV`|[PanoFormer: Panorama Transformer for Indoor 360 Depth Estimation](https://arxiv.org/pdf/2203.09283)|[![Github stars](https://img.shields.io/github/stars/zhijieshen-bjtu/PanoFormer.svg)](https://github.com/zhijieshen-bjtu/PanoFormerr)|---|
|2022|`AAAI`|[Improving 360 monocular depth estimation via non-local dense prediction transformer and joint supervised and self-supervised learning](https://arxiv.org/pdf/2109.10563)|---|---| 
|2022|`arXiv`|[MVSFormer: Multi-view stereo by learning robust image features and temperature-based depth](https://arxiv.org/pdf/2208.02541)|---|---|
|2022|`arXiv`|[Objcavit: improving monocular depth estimation using natural language models and image-object cross-attention](https://arxiv.org/pdf/2211.17232)|[![Github stars](https://img.shields.io/github/stars/DylanAuty/ObjCAViT.svg)](https://github.com/DylanAuty/ObjCAViT)|---|
|2022|`arXiv`|[Depthformer: Multiscale Vision Transformer For Monocular Depth Estimation With Local Global Information Fusion](https://arxiv.org/pdf/2207.04535)|[![Github stars](https://img.shields.io/github/stars/ashutosh1807/Depthformer.svg)](https://github.com/ashutosh1807/Depthformer)|---|
|2022|`arXiv`|[Sidert: A real-time pure transformer architecture for single image depth estimation](https://arxiv.org/pdf/2204.13892)|---|---|
|2022|`ECCV`|[Hybrid transformer based feature fusion for self-supervised monocular depth estimation](https://arxiv.org/pdf/2211.11066)|---|---|
|2022|`ECCV`|[Spike transformer: Monocular depth estimation for spiking camera](https://fq.pkwyx.com/default/https/www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670034.pdf)|[![Github stars](https://img.shields.io/github/stars/Leozhangjiyuan/MDE-SpikingCamera.svg)](https://github.com/Leozhangjiyuan/MDE-SpikingCamera)|---|
|2022|`3DV`|[MonoViT: Self-Supervised Monocular Depth Estimation with a Vision Transformer](https://arxiv.org/pdf/2208.03543)|[![Github stars](https://img.shields.io/github/stars/zxcqlf/MonoViT.svg)](https://github.com/zxcqlf/MonoViT)|---|
|2022|`arXiv`|[DEST: "Depth Estimation with Simplified Transformer](https://arxiv.org/pdf/2204.13791)|---|---|
|2022|`arXiv`|[SparseFormer: Attention-based Depth Completion Network](https://arxiv.org/pdf/2206.04557)|---|---|
|2022|`CVPR`|[GuideFormer: Transformers for Image Guided Depth Completion](https://openaccess.thecvf.com/content/CVPR2022/papers/Rho_GuideFormer_Transformers_for_Image_Guided_Depth_Completion_CVPR_2022_paper.pdf)|---|---|
|2022|`CVPR`|[Multi-frame self-supervised depth with transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Guizilini_Multi-Frame_Self-Supervised_Depth_With_Transformers_CVPR_2022_paper.pdf)|---|---|
|2022|`arXiv`|[Transformers in Self-Supervised Monocular Depth Estimation with Unknown Camera Intrinsics](https://arxiv.org/pdf/2202.03131)|---|---|
|2021|`ICCV`|[Revisiting stereo depth estimation from a sequence-to-sequence perspective with transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Revisiting_Stereo_Depth_Estimation_From_a_Sequence-to-Sequence_Perspective_With_Transformers_ICCV_2021_paper.pdf)|---|STTR<br>stereo matching|
|2021|`BMVC`|[Transformer-based Monocular Depth Estimation with Attention Supervision](https://www.bmvc2021-virtualconference.com/assets/papers/0244.pdf)|[![Github stars](https://img.shields.io/github/stars/WJ-Chang-42/ASTransformer.svg)](https://github.com/WJ-Chang-42/ASTransformer)|---|
|2021|`ICCV`|[Transformer-Based Attention Networks for Continuous Pixel-Wise Prediction](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Transformer-Based_Attention_Networks_for_Continuous_Pixel-Wise_Prediction_ICCV_2021_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/ygjwd12345/TransDepth.svg)](https://github.com/ygjwd12345/TransDepth)|---|
|2021|`ICCV`|[Vision transformers for dense prediction](https://openaccess.thecvf.com/content/ICCV2021/papers/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/isl-org/DPT.svg)](https://github.com/isl-org/DPT)|DPT|

<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->

## Other Resources
* [Awesome-Transformer-Attention](https://github.com/cmhungsteve/Awesome-Transformer-Attention)
* [Dense-Prediction-Transformer-Based-Visual-Odometry](https://github.com/sumedhreddy90/Dense-Prediction-Transformer-Based-Visual-Odometry)
* [Visual SLAM with Vision Transformers(ViT)](https://github.com/MisterEkole/slam_with_vit)
* [Awesome-Learning-based-VO-VIO](https://github.com/KwanWaiPang/Awesome-Learning-based-VO-VIO)
* Some basic paper in ViT:

| Year | Venue | Paper Title | Repository | Note |
|:----:|:-----:| ----------- |:----------:|:----:|
|2024|`Transactions on Machine Learning Research Journal`|[Dinov2: Learning robust visual features without supervision](https://arxiv.org/pdf/2304.07193)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/dinov2.svg)](https://github.com/facebookresearch/dinov2)|DINO2|
|2021|`ICML`|[Is space-time attention all you need for video understanding?](https://arxiv.org/pdf/2102.05095)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/TimeSformer.svg)](https://github.com/facebookresearch/TimeSformer)|TimeSformer|
|2021|`CVPR`|[Taming transformers for high-resolution image synthesis](https://openaccess.thecvf.com/content/CVPR2021/papers/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/CompVis/taming-transformers.svg)](https://github.com/CompVis/taming-transformers)|High resolution CNN+Transformer|
|2021|`ICCV`|[Emerging properties in self-supervised vision transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/facebookresearch/dino.svg)](https://github.com/facebookresearch/dino)|DINO<br>SSL| 
|2021|`ICCV`|[Vivit: A video vision transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.pdf)|[![Github stars](https://img.shields.io/github/stars/lucidrains/vit-pytorch.svg)](https://github.com/lucidrains/vit-pytorch)|---|
|2020|`ICLR`|[An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/pdf/2010.11929/1000)|[![Github stars](https://img.shields.io/github/stars/google-research/vision_transformer.svg)](https://github.com/google-research/vision_transformer)|ViT|


<!-- |---|`arXiv`|---|---|---| -->
<!-- [![Github stars](https://img.shields.io/github/stars/***.svg)]() -->





