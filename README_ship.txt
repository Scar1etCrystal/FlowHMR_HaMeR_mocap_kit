FlowHaMeR Windows组合包——SAM N4线性插值 + Fast + HaMeR N2 + query分批注意力

日常入口：把视频拖到 Video_to_FBX.bat，自动得到合并后的final.fbx。
默认手部条件：SAM双手每4帧真实计算一次，左右手分别对相邻观测做线性插值。
默认attention_backend=query_batched，每批256个query，refiner仍全局。
该一键入口默认单段上限3600帧；original注意力回退仍限2400帧。
完整一键用法与限制请读 README_一键视频到FBX.txt。
默认参数在 pipeline_settings.json；各阶段独立进程、缓存校验、中文路径支持。
下面第2、4步的单阶段脚本已移到 tools\diagnostics\，只用于排查，不是入口。


位置：本仓库根目录（包根）。下文相对路径都相对包根。
环境与权重：WEIGHTS_AND_ENV.md、weights_manifest.json
上游安装与资源布局：app\flowhmr\docs\install.md、app\flowhmr\docs\resources.md
包内说明：README_一键视频到FBX.txt（日常使用）、app\flowhmr\README.md
（本地工作与实验留档不随仓库发布。）

1. 自检
  envs\python.exe tools\check_env.py

2. FlowHMR单阶段（内部诊断，输出app\flowhmr\output\<视频名>\flowhmr_latest.npz/.fbx）
  tools\diagnostics\run_flowhmr.bat "视频.mp4" --vggt_interval 10 --postprocess

当前分阶段驻留：相机前将其它模型移CPU、相机后释放VGGT、无缓存时才加载SAM提特征、生成前释放SAM恢复主网络。
不会改权重/精度/注意力/ODE，不把视频分块。

12GiB/FP32/20步/CFG=1：建议一次2400帧（转码30fps的80秒），2700帧成功但贴边。
长视频相机也要控制为约60张抽样；2400帧用 --vggt_interval 41。
固定10约600帧已有61张相机抽样。这个预算随画幅、桌面显存、CFG而变，不保证任意视频都同一个上限。

3. 生成主角框供手部使用
  envs\python.exe tools\make_main_track.py --bbox "app\flowhmr\output\<视频名>\<视频名>_bbox.npz" --output "main_track.npz"

4. HaMeR单阶段（内部诊断，已改默认N=2，TTA=1）
  tools\diagnostics\run_hamer.bat "app\flowhmr\output\<视频名>\<视频名>_30fps.mp4" "main_track.npz" "手输出目录"
  可加 --detect_interval 1 或3；其它参数见新文档。

只在f%N==0运行ViTPose和HaMeR；输出原始旋转SLERP插值后完整序列平滑。
输出 <stem>_hamer_data.pkl + sampled_raw.pkl + <stem>_sampling.npz + run_info.json。
严格用FlowHMR同一个转码视频，不能拿原24fps视频对30fps身体结果。

5. 合并及FBX（仅替换30个指关节，保留身体/手腕/根）
  envs\python.exe tools\merge_hamer_smplh.py --body "身体.npz" --hands "手.pkl" --output "最终.fbx"
  同名NPZ一起保存；左手M@R@M只做一次；帧数不一致就报错。

产物已FBX SDK重新导入验证：52骨骼、1蒙皮网格、帧数与NPZ一致；未做Blender视觉验收。
原FBX逐帧Euler无unwrap缺陷仍未修。

6. 重要结果修订
  真正N2推理18.90秒 vs逐帧32.25秒；含加载/导出32.68 vs45.06秒。
  手指相对逐帧平均差3.578度、p99 14.647、最大23.756。
  旧0.747度来自已平滑序列抽样，旧68/N不是实测，请勿继续据此估计。

7. 已知事项
  - 去滤波只跳FlowHMR普通3帧平滑；其它IK/异常旋转/轨迹平滑保留。HaMeR仍window11独立平滑。
  - 地面流程以脚接触为主，打滚时没有背/肩接触防浮空项；尚未修改该算法，不能把跳跃逐帧强制吸地。
  - ViTPose现配置加载wholebody权重报告unexpected expert keys，需另外审查绝对关键点质量；本次N1/N2在相同配置下对照。
  - 旧chunk报告只是另一段素材的本地对照，有相机/跟踪等混杂因素，现不走分块方案。
  - 原PoseCapture_Pack安装未删除；重复下载权重已校验删除10.96GiB。
