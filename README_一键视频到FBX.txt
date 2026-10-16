FlowHaMeR 组合包：拖拽视频 → 合并FBX
===================================

使用方法
--------
把视频文件拖到包根的 Video_to_FBX.bat 上，等待出现 DONE。
可以一次拖多个视频，会依次运行。双击BAT但没拖视频会显示提示。

结果在：
  outputs\video_to_fbx\视频名_标识\final.fbx
  outputs\video_to_fbx\视频名_标识\final.npz

同目录有 working_30fps.mp4，和最终FBX逐帧对应；Blender帧率设30。
每个阶段的日志、采样帧号、相机间隔、耗时和最终SDK验证在同一目录。
运行失败窗口不会自动消失，请查看末尾错误及对应阶段.log。

npz 字段
--------
final.npz：
- poses (T,52,3)：SMPL-H轴角，手指已合入HaMeR。
- trans (T,3)：根平移，米。
- betas (1,16)、Rh (T,3)、num_frames、mocap_framerate、gender。
- foot_contacts (T,4) float32：逐帧脚接触分数，列顺序
  ['L_Ankle','L_Foot','R_Ankle','R_Foot']。
  这是模型输出的概率，后处理用它按 >0.7 判定接触段。
  它与poses/trans的Y轴180度旋转无关，不需要跟着旋转。

同目录另有两份身体结果，字段与final.npz相同、都含foot_contacts：
- body_origin.npz：FlowHMR网络原始输出，未做后处理，手指也是FlowHMR的。
- body_postprocess.npz：做完脚接触/IK等后处理，手指仍是FlowHMR的。
对应上游的 <name>_origin.* 与 <name>.* 两份。
final.npz = body_postprocess.npz 再替换30个手指关节。

字段名与上游FlowHMR保存路径一致，可直接沿用原有下游脚本。

本次选定的默认配置
------------------
- Fast-SAM-3D-Body，SKIP_KEYPOINT_PROMPT=1。
- 身体参考信息逐帧计算。
- SAM左右手每60帧一组，用固定种子在每组随机取1帧，再分别取全片左右手均值。
  这是已验收的N60平均参考，不是之前SAM手N2动态插值。
- HaMeR只在 f % 2 == 0 推理；其余帧用输出姿态SLERP补齐，TTA=1。
- FlowHMR 20步、seed0、CFG1。
- 默认普通PyTorch query分批注意力，每批256；主干原局部可见规则不变，
  token_refiner仍看全片全部有效K/V。没有按片段独立生成或拼接姿态。
- 开启FlowHMR postprocess，POSTPROCESS_SKIP_SMOOTH=1。
- 最后把HaMeR的30个指关节合入SMPL-H，左手镜像一次。
  身体、手腕、根平移由FlowHMR保留。

“无滤波postprocessing”的准确含义
---------------------------------
沿用此前约定：取消FlowHMR普通三帧平滑，保留脚接触、异常旋转修补和IK。
HaMeR仍沿用已验收的window11手部平滑，与身体去滤波开关独立。
没有修改此前搁置的地面算法。

显存与输入约束
--------------
模型分阶段在独立进程运行，结束后释放显存。
VGGT默认按视频长度自动调相机间隔，目标不超过约31张采样图（最小间隔10）；
适合目前的固定机位单镜头素材。快速移动镜头需另验相机效果。

当前仍为整段生成，没有加入之前暂缓的动作分块。
query分批版默认上限3600帧（30fps约两分钟）；已用3789帧缓存验证同一生产
worker的完整生成+后处理，峰值allocated分别约2.83/4.70GiB。
这不是3789帧全部预处理、HaMeR、导出均已冷启动的保证；更长视频还要看CPU内存
与HaMeR缓存等资源。超过默认上限会明确报错，不截断。
切回original注意力时，12GB preset仍自动限制到2400帧（约80秒）。

建议画面内单人、全身尽量可见、固定机位、单镜头。
自动选轨迹有启发式合并，并不保证复杂多人场景的身份选择。

帧对齐
------
输入统一转为30fps。输出覆盖整个转码视频，身体与手部使用同一帧号。
人物轨迹中间缺帧插值，首尾未跟踪段沿用最近人体框，不会静默丢帧；
实际观测范围写入video_info.json。长时间人物离开画面无法保证动作有效。

缓存与重跑
----------
输入视频内容、模式、N、脚本和权重信息组成任务标识。
再次拖同一视频会校验中间文件后复用；改配置生成新的任务目录。
旧实验/旧逐帧特征不会拿来冒充N60加速。
需要完全新跑可用：
  Video_to_FBX.bat "视频路径" --fresh

sam_hand_interval调的是估均值时的采样间隔；设为1仍是全帧求均值，不等于恢复逐帧动态手部条件。

已完成的验证
------------
新版query分批整链：一段实际1499帧、1920x1080/30fps素材。
真实BAT从视频到最终FBX全冷跑903.71秒（15分04秒）。
产物：<包根>\outputs\video_to_fbx\<视频名>_<标识>\final.fbx。
同目录working_30fps.mp4与FBX逐帧对齐；MKV元数据1500帧但实际解码1499，无额外丢帧。
新旧attention同输入warm ABBA：生成20.55→11.20秒，快1.83倍；生成峰值4.10→2.24GiB。
整链仍由SAM/HaMeR/后处理占大头，估计总时间仅省约5–9秒，不是整链快1.83倍。
当前整链torch峰值在VGGT，7.25GiB；显卡含桌面/驱动采样峰值约8.49GiB。
最终FBX重新导入1499帧、52骨骼、1网格；新旧身体平均差.0085°、根最大差.447mm。
新模式缓存续跑六阶段全部命中，original回退实际生成/合并也已跑通。
新版详情：输出目录内的 generation_info.json、run_manifest.json 与各阶段 .log。

历史首版（原attention）验证：
中文、空格、&文件名通过真实BAT全冷启动测试。151帧总耗时约264秒。
SAM身体前向151次、双手3次；HaMeR ViTPose76次、手裁剪152个。
最终FBX已SDK重新导入，151帧、52骨架、1网格，关键帧与NPZ一致。
同文件再次拖入经hash校验会复用缓存，批量续跑测试已通过。

可调设置
--------
包根 pipeline_settings.json 集中设置 SAM N、HaMeR N、批大小、手部平滑窗口、
FlowHMR步数、相机采样预算、线程数与单段上限。
非程序用户保持默认即可。
attention_backend="query_batched" 为新默认，attention_query_chunk=256。
需回退可把attention_backend改为"original"，不需要删除新模块。
切模式或改query批大小会产生新的任务标识，不复用另一种注意力的生成结果。
query批大小调的是一层内的执行批次，不是视频生成分块长度。
settings.bat保留Python/模型离线路径等环境设置。

已有独立入口 run_flowhmr.bat、run_hamer.bat 保留作诊断，
日常整链请只用 Video_to_FBX.bat。

关于N60
-------
SAM手部参考信息用每60帧一组的抽样均值。验收时它相对逐帧手条件下的根
平移漂移明显大于隔2帧方案，用户人工确认该观感可接受后，本包按此采用。
如需更贴近逐帧手条件，可调小 sam_hand_interval（采样帧会变多）。

已知导出行为
------------
FBX沿用原SMPL-H导出器，52骨架+蒙皮网格，米转厘米。
逐帧Euler未unwrap的问题仍保留；SDK会验证整数关键帧与NPZ一致，
未把这个验证当成Blender导入器的帧间插值视觉验收。
