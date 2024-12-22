import re

log_data = """
2024-12-21 23:01:29,765 - jaredLogger - INFO - iter 107000: loss 0.9828, time 28559.65ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:01:35,597 - jaredLogger - INFO - iter 107010: loss 0.5905, time 291.12ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:01:41,435 - jaredLogger - INFO - iter 107020: loss 0.5505, time 291.98ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:01:47,275 - jaredLogger - INFO - iter 107030: loss 0.5462, time 292.18ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:01:53,100 - jaredLogger - INFO - iter 107040: loss 0.5295, time 292.29ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:01:58,936 - jaredLogger - INFO - iter 107050: loss 0.5426, time 292.02ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:04,765 - jaredLogger - INFO - iter 107060: loss 0.5768, time 292.18ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:10,597 - jaredLogger - INFO - iter 107070: loss 0.5481, time 291.15ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:16,433 - jaredLogger - INFO - iter 107080: loss 0.5634, time 290.59ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:22,267 - jaredLogger - INFO - iter 107090: loss 0.5791, time 292.14ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:28,104 - jaredLogger - INFO - iter 107100: loss 0.5178, time 291.94ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:33,936 - jaredLogger - INFO - iter 107110: loss 0.5523, time 291.23ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:39,762 - jaredLogger - INFO - iter 107120: loss 0.5596, time 292.28ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:45,589 - jaredLogger - INFO - iter 107130: loss 0.4966, time 291.80ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:51,425 - jaredLogger - INFO - iter 107140: loss 0.5369, time 292.07ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:02:57,257 - jaredLogger - INFO - iter 107150: loss 0.5130, time 290.98ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:03,091 - jaredLogger - INFO - iter 107160: loss 0.5194, time 292.23ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:08,932 - jaredLogger - INFO - iter 107170: loss 0.5267, time 291.91ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:14,759 - jaredLogger - INFO - iter 107180: loss 0.5525, time 291.45ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:20,596 - jaredLogger - INFO - iter 107190: loss 0.5301, time 292.30ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:26,424 - jaredLogger - INFO - iter 107200: loss 0.5439, time 291.61ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:32,263 - jaredLogger - INFO - iter 107210: loss 0.5295, time 291.54ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:38,105 - jaredLogger - INFO - iter 107220: loss 0.5301, time 292.34ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:43,947 - jaredLogger - INFO - iter 107230: loss 0.5531, time 294.84ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:49,785 - jaredLogger - INFO - iter 107240: loss 0.4837, time 291.73ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:03:55,622 - jaredLogger - INFO - iter 107250: loss 0.5081, time 292.33ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:01,451 - jaredLogger - INFO - iter 107260: loss 0.5151, time 291.85ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:07,283 - jaredLogger - INFO - iter 107270: loss 0.5364, time 291.35ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:13,115 - jaredLogger - INFO - iter 107280: loss 0.5396, time 293.22ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:18,961 - jaredLogger - INFO - iter 107290: loss 0.5464, time 291.67ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:24,797 - jaredLogger - INFO - iter 107300: loss 0.5081, time 291.48ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:30,629 - jaredLogger - INFO - iter 107310: loss 0.5693, time 291.32ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:36,462 - jaredLogger - INFO - iter 107320: loss 0.5679, time 292.38ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:42,296 - jaredLogger - INFO - iter 107330: loss 0.5568, time 290.19ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:48,124 - jaredLogger - INFO - iter 107340: loss 0.5882, time 291.16ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:53,952 - jaredLogger - INFO - iter 107350: loss 0.5830, time 291.27ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:04:59,787 - jaredLogger - INFO - iter 107360: loss 0.5441, time 291.50ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:05,613 - jaredLogger - INFO - iter 107370: loss 0.5434, time 291.70ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:11,449 - jaredLogger - INFO - iter 107380: loss 0.5075, time 293.54ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:17,280 - jaredLogger - INFO - iter 107390: loss 0.5210, time 292.05ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:23,115 - jaredLogger - INFO - iter 107400: loss 0.5738, time 291.14ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:28,951 - jaredLogger - INFO - iter 107410: loss 0.5399, time 291.23ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:34,777 - jaredLogger - INFO - iter 107420: loss 0.5338, time 291.22ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:40,610 - jaredLogger - INFO - iter 107430: loss 0.5321, time 292.26ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:46,447 - jaredLogger - INFO - iter 107440: loss 0.5484, time 291.14ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:52,277 - jaredLogger - INFO - iter 107450: loss 0.5085, time 291.39ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:05:58,115 - jaredLogger - INFO - iter 107460: loss 0.5296, time 291.09ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:03,945 - jaredLogger - INFO - iter 107470: loss 0.4882, time 291.97ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:09,777 - jaredLogger - INFO - iter 107480: loss 0.5196, time 290.37ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:15,613 - jaredLogger - INFO - iter 107490: loss 0.5562, time 291.37ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:21,449 - jaredLogger - INFO - iter 107500: loss 0.5521, time 291.57ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:27,281 - jaredLogger - INFO - iter 107510: loss 0.5444, time 291.78ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:33,112 - jaredLogger - INFO - iter 107520: loss 0.5407, time 291.26ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:38,942 - jaredLogger - INFO - iter 107530: loss 0.5343, time 292.20ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:44,779 - jaredLogger - INFO - iter 107540: loss 0.5286, time 291.96ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:50,614 - jaredLogger - INFO - iter 107550: loss 0.5529, time 290.90ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:06:56,446 - jaredLogger - INFO - iter 107560: loss 0.5213, time 291.01ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:02,281 - jaredLogger - INFO - iter 107570: loss 0.5385, time 291.82ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:08,114 - jaredLogger - INFO - iter 107580: loss 0.5075, time 291.77ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:13,947 - jaredLogger - INFO - iter 107590: loss 0.5561, time 290.84ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:19,779 - jaredLogger - INFO - iter 107600: loss 0.4734, time 291.43ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:25,621 - jaredLogger - INFO - iter 107610: loss 0.5474, time 292.42ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:31,453 - jaredLogger - INFO - iter 107620: loss 0.4798, time 291.93ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:37,291 - jaredLogger - INFO - iter 107630: loss 0.5086, time 292.66ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:43,128 - jaredLogger - INFO - iter 107640: loss 0.5203, time 291.47ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:48,957 - jaredLogger - INFO - iter 107650: loss 0.5132, time 291.44ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:07:54,799 - jaredLogger - INFO - iter 107660: loss 0.5269, time 291.50ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:08:00,640 - jaredLogger - INFO - iter 107670: loss 0.5696, time 291.42ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
2024-12-21 23:08:06,473 - jaredLogger - INFO - iter 107680: loss 0.5433, time 292.27ms, seq_acc: 0.00%, gt. conf: 1.0000, ans. conf: 1.0000 diff: 0.0000
"""
# Extract loss values using regex
loss_values = [float(match.group(1)) for match in re.finditer(r"loss (\d+\.\d+)", log_data)]

# Calculate the mean loss
mean_loss = sum(loss_values) / len(loss_values)

print(f"Mean Loss: {mean_loss:.4f}")
