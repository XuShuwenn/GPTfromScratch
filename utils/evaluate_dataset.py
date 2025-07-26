import os
import collections
import re

from datasets import load_from_disk
from tqdm import tqdm

# 英文停用词列表（手动定义，避免NLTK依赖）
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
    'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 
    'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', 
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
    'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def split_sentences(text):
    """使用正则表达式分割句子"""
    sentence_endings = re.compile(r'[.!?]\s+')
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]  # 去除空白句子

def simple_pos_tag(words):
    """简单的词性标注，识别名词（以基本规则为准）"""
    nouns = []
    # 简单的名词识别规则：
    # 1. 以 -tion, -sion, -ness, -ment, -ing, -er, -or, -ar, -ty, -ity 结尾的词
    # 2. 不在停用词中的词
    noun_suffixes = ['tion', 'sion', 'ness', 'ment', 'ing', 'er', 'or', 'ar', 'ty', 'ity', 'al', 'ism']
    
    for word in words:
        if word not in ENGLISH_STOPWORDS:
            # 检查词尾
            if any(word.endswith(suffix) for suffix in noun_suffixes):
                nouns.append(word)
            # 长度大于3且不是常见的动词、形容词等
            elif len(word) > 3 and word not in ['said', 'went', 'came', 'good', 'nice', 'big', 'small', 'old', 'new']:
                nouns.append(word)
    
    return nouns

def tokenize_words(text):
    """使用正则表达式分割单词"""
    word_pattern = re.compile(r'\b\w+\b')
    return word_pattern.findall(text.lower())

def analyze_dataset(dataset_path):
    """
    分析数据集的各项指标：句子复杂性、词汇多样性、领域多样性等。
    """
    print(f"正在从 {dataset_path} 加载数据集...")
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return

    # 加载数据集的训练部分
    try:
        dataset = load_from_disk(dataset_path)['validation']
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    print("数据集加载成功，开始分析...")

    # 初始化统计变量
    total_stories = 0
    total_sentences = 0
    total_words = 0
    all_words = []
    
    # 遍历数据集
    for story in tqdm(dataset, desc="处理故事中"):
        text = story['text']
        total_stories += 1
        
        # 句子分析
        sentences = split_sentences(text)
        num_sentences = len(sentences)
        total_sentences += num_sentences
        
        # 单词分析
        words = tokenize_words(text)
        all_words.extend(words)
        total_words += len(words)

    # 1. 基本统计
    avg_sentences_per_story = total_sentences / total_stories if total_stories > 0 else 0

    # 2. 句子复杂性
    avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0

    # 3. 词汇多样性
    # 过滤掉标点符号
    words_alpha = [word for word in all_words if word.isalpha()]
    total_alpha_words = len(words_alpha)
    vocab = set(words_alpha)
    vocab_size = len(vocab)
    ttr = vocab_size / total_alpha_words if total_alpha_words > 0 else 0 # Type-Token Ratio

    # 4. 领域多样性 (通过最常见的名词来近似)
    words_filtered = [word for word in words_alpha if word not in ENGLISH_STOPWORDS]
    
    # 词性标注，找到名词
    nouns = simple_pos_tag(words_filtered)
    noun_freq = collections.Counter(nouns)
    most_common_nouns = noun_freq.most_common(20)

    # 打印报告
    print("\n" + "="*50)
    print(" TinyStories 数据集评估报告")
    print("="*50)
    print("\n### 1. 基本统计 ###")
    print(f"  - 故事总数: {total_stories:,}")
    print(f"  - 句子总数: {total_sentences:,}")
    print(f"  - 单词总数 (包含标点): {total_words:,}")
    print(f"  - 平均每篇故事句子数: {avg_sentences_per_story:.2f}")

    print("\n### 2. 句子复杂性 ###")
    print(f"  - 平均句长 (单词数): {avg_words_per_sentence:.2f}")

    print("\n### 3. 词汇多样性 ###")
    print(f"  - 总词汇量 (独立单词数): {vocab_size:,}")
    print(f"  - 词汇密度 (Type-Token Ratio): {ttr:.4f}")
    print("    (注: TTR 越接近 1，词汇多样性越高)")

    print("\n### 4. 领域多样性 (Top 20 核心名词) ###")
    for i, (noun, count) in enumerate(most_common_nouns):
        print(f"  {i+1:2d}. {noun:<15} - {count:,} 次")
    print("\n" + "="*50)


if __name__ == "__main__":
    # 数据集路径
    # 假设脚本在 utils/ 目录下，数据集在项目根目录的 data/tinystories
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    dataset_path = os.path.join(project_root, 'data', 'tinystories')
    
    analyze_dataset(dataset_path)