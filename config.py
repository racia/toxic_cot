#data_path
babi_train_data_path = './data/qa{}_train.jsonl'

csqa_train_data_path = './data/CommonsenseQA/train_rand_split.jsonl'
csqa_dev_data_path = './data/CommonsenseQA/dev_rand_split.jsonl'

wino_train_data_path = './data/winogrande_1.1/train_l.jsonl'
wino_dev_data_path = './data/winogrande_1.1/dev.jsonl'

hella_train_data_path = './data/hellaswag/hellaswag_train.jsonl'
hella_dev_data_path = './data/hellaswag/hellaswag_val.jsonl'

siqa_train_data_path = './data/SocialIQA/train.jsonl'
siqa_train_label_path = './data/SocialIQA/train-labels.lst'
siqa_dev_data_path = './data/SocialIQA/dev.jsonl'
siqa_dev_label_path = './data/SocialIQA/dev-labels.lst'

piqa_train_data_path = './data/PIQA/train.jsonl'
piqa_train_label_path = './data/PIQA/train-labels.lst'
piqa_dev_data_path = './data/PIQA/valid.jsonl'
piqa_dev_label_path = './data/PIQA/valid-labels.lst'

gsm8k_train_data_path = './data/grade-school-math/grade_school_math/data/train.jsonl'
gsm8k_dev_data_path = './data/grade-school-math/grade_school_math/data/test.jsonl'

strategy_data_path = './data/strategyqa_dataset/strategyqa_train_filtered.jsonl'

#W&C_index
#wino c2w 180
wino_c2w_index = [4, 7, 13, 15, 18, 27, 36, 40, 41, 47, 50, 53, 60, 69, 71, 73, 76, 80, 84, 97, 100, 108, 113, 114, 119, 121, 132, 151, 158, 160, 171, 175, 180, 183, 185, 189, 197, 199, 201, 206, 207, 209, 232, 235, 245, 253, 255, 266, 272, 274, 284, 285, 292, 306, 307, 316, 320, 323, 327, 333, 338, 342, 347, 381, 387, 390, 393, 407, 409, 418, 423, 426, 427, 433, 439, 444, 453, 454, 455, 459, 467, 473, 475, 478, 479, 481, 482, 490, 493, 498, 512, 518, 525, 529, 531, 535, 538, 543, 557, 560, 568, 573, 574, 580, 582, 595, 597, 600, 605, 610, 620, 627, 638, 640, 646, 654, 661, 666, 677, 678, 686, 689, 693, 695, 710, 711, 712, 714, 721, 733, 735, 739, 740, 745, 752, 753, 759, 760, 766, 768, 772, 774, 776, 780, 782, 798, 808, 819, 824, 831, 836, 842, 848, 849, 861, 868, 869, 872, 873, 882, 893, 903, 911, 916, 920, 927, 928, 930, 943, 960, 962, 967, 973, 976, 977, 979, 981, 984, 995, 997]
#wino w2c 177
wino_w2c_index = [1, 2, 20, 22, 23, 32, 34, 38, 42, 46, 54, 59, 62, 65, 77, 79, 81, 89, 91, 96, 99, 101, 103, 104, 112, 122, 124, 127, 142, 144, 145, 170, 179, 182, 190, 191, 198, 205, 210, 212, 213, 215, 229, 239, 243, 247, 256, 265, 276, 279, 291, 294, 297, 303, 311, 328, 335, 348, 349, 353, 356, 358, 360, 370, 371, 372, 379, 380, 384, 388, 401, 405, 414, 435, 437, 441, 442, 443, 452, 458, 462, 464, 470, 474, 484, 491, 505, 507, 509, 510, 514, 517, 520, 523, 528, 532, 534, 544, 550, 554, 555, 564, 566, 576, 585, 591, 603, 606, 617, 621, 623, 629, 633, 639, 641, 644, 648, 655, 663, 665, 672, 681, 685, 688, 691, 697, 702, 707, 719, 743, 744, 747, 748, 750, 763, 767, 778, 790, 803, 816, 820, 825, 826, 829, 830, 841, 843, 853, 856, 871, 875, 876, 880, 889, 891, 897, 899, 901, 902, 904, 907, 909, 910, 914, 915, 921, 941, 942, 944, 948, 956, 958, 965, 972, 974, 980, 993]
#csqa c2w 75
csqa_c2w_index = [10, 24, 36, 41, 49, 137, 149, 158, 161, 174, 177, 193, 219, 220, 231, 244, 276, 283, 286, 297, 303, 308, 331, 340, 355, 379, 386, 394, 395, 402, 413, 424, 431, 441, 443, 450, 457, 467, 488, 521, 523, 525, 527, 539, 599, 604, 645, 652, 654, 685, 700, 709, 738, 754, 770, 795, 825, 826, 858, 869, 881, 893, 898, 903, 910, 913, 925, 929, 930, 939, 940, 946, 955, 993, 998]
#csqa w2c 114
csqa_w2c_index = [2, 5, 7, 14, 26, 31, 34, 35, 48, 58, 66, 75, 92, 96, 103, 109, 122, 125, 126, 127, 175, 184, 185, 186, 191, 200, 209, 218, 245, 247, 248, 249, 250, 253, 260, 267, 274, 293, 295, 314, 322, 324, 356, 363, 364, 370, 376, 380, 385, 387, 398, 412, 429, 438, 446, 513, 516, 524, 532, 543, 550, 566, 567, 588, 590, 592, 593, 594, 601, 602, 607, 616, 622, 624, 628, 633, 639, 640, 644, 646, 659, 673, 705, 713, 718, 721, 723, 744, 747, 755, 756, 758, 760, 768, 771, 776, 781, 791, 805, 818, 827, 829, 832, 835, 845, 851, 860, 892, 895, 927, 932, 956, 957, 972]
OPENAI_API_KEY = 'fGBdIFoaDUeLQ'
# OPENAI_API_KEY = 'sk-i7t4FKCdavAisTCWFc2f9737854348F29d17C9E7De2e9d9e'
max_requests_per_minute = 3500 # 3_000 * 0.5
max_tokens_per_minute = 90000 #250_000 * 0.5

# max_requests_per_minute = 60 # 3_000 * 0.5
# max_tokens_per_minute = 60000 #250_000 * 0.5
# request_url = "https://api.openai.com/v1/chat/completions"
request_url = 'https://ai.liaobots.work/v1/chat/completions'
# request_url = 'https://api.xty.app/v1/chat/completions'
