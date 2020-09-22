"""最终的文章处理"""
import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs
import re
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle


def load_data():
    import csv
    csv_file = open('./dataC/traintext.csv', encoding='utf-8')
    csv_reader_lines = csv.reader(csv_file)
    for one_line in csv_reader_lines:
        yield one_line


def cut_word(text, stopwords_list):
    # 分词函数
    def execute_cut_text(sentence):
        seg_list = pseg.lcut(sentence)
        seg_list = [i for i in seg_list if i.flag not in stopwords_list]
        filtered_words_list = []
        for seg in seg_list:
            if len(seg.word) <= 1:
                continue
            elif seg.flag == "eng":
                if len(seg.word) <= 2:
                    continue
                else:
                    filtered_words_list.append(seg.word)
            elif seg.flag.startswith("n"):
                filtered_words_list.append(seg.word)
            elif seg.flag in ["x", "eng"]:  # 是自定一个词语或是英文单词
                filtered_words_list.append(seg.word)
        return filtered_words_list

    text_before_processing = re.sub("<.*?>", "", text)
    text_word_segments = execute_cut_text(text_before_processing)
    text_word_segments_to_str = ' '.join(text_word_segments)
    return text_word_segments_to_str


def get_text_rank(text, stopwords_list):
    # TextRank函数
    class TextRank(jieba.analyse.TextRank):
        def __init__(self, window=20, word_min_len=2):
            super(TextRank, self).__init__()
            self.span = window
            self.word_min_len = word_min_len
            # 需要保留的词性
            self.pos_filt = frozenset(
                ('n', 'eng', 'f', 's', 't', 'nr', 'ns', 'nw', 'nz', 'PER')
            )

        def pair_filter(self, wp):
            # 过滤条件
            if wp.flag == "eng":
                if len(wp.word) <= 2:
                    return False
            if wp.flag in self.pos_filt and len(wp.word.strip()) >= self.word_min_len and wp.word.lower() not in stopwords_list:
                return True

    text_rank_model = TextRank(window=5, word_min_len=2)
    allow_pos = ('n', 'eng', 'nr', 'ns', 'nt', 'nw', 'nz', 'c')
    tags = text_rank_model.textrank(text, topK=25, withWeight=True, allowPOS=allow_pos)
    return tags


def get_cv_tf_idf(text, cv_vocabulary, tf_idf_model):
    text_array = [text]
    text_tf_idf = tf_idf_model.transform(cv_vocabulary.transform(text_array))
    return text_tf_idf.toarray()


def get_text_vector(text_rank, tf_idf, vocabulary):

    def get_index_of_vocabulary(vocabulary, key_word, candidate_key_words_list):
        whether_candidate = False
        if not vocabulary.get(key_word):
            print(key_word+" 在TF词典中找不到。")
            print("启用候选词")
            whether_candidate = True
            # 取候选列表中的第一个元组作为候选元组
            for candidate_tuple_index, candidate_tuple in enumerate(candidate_keywords_list):
                candidate_key_word = candidate_tuple[0].lower()
                if vocabulary.get(candidate_key_word):
                    candidate_keywords_list.pop(candidate_tuple_index)
                    print("找到的候选词为:" + candidate_key_word)
                    repertory = vocabulary[candidate_key_word]
                    candidate_word_text_rank = candidate_tuple[1]
                    return candidate_key_word, repertory, candidate_keywords_list, whether_candidate, \
                           candidate_word_text_rank
            # 找完了候选单词都找不到
            print(key_word + " 在TF词典中找不到。且没有可供候选的关键词")
            return 0
        else:
            repertory = vocabulary[key_word]
            candidate_word_text_rank = 0
            return key_word, repertory, candidate_key_words_list, whether_candidate, candidate_word_text_rank
    text_vector = []
    if not text_rank:
        print("改文章没有关键词列表")
        text_vector = 20 * [0]
    # 根据text_rank区分主关键词和备用关键词
    ratio = 0.8   # 切分比
    execute_offset = int(len(text_rank) * ratio)
    keywords_list = text_rank[:execute_offset]
    print("该交互信息内容的主关键词为: " + str(keywords_list))
    candidate_keywords_list = text_rank[execute_offset:]
    print("该交互信息内容的备用关键词为: " + str(candidate_keywords_list))
    for tuples in keywords_list:
        keyword = tuples[0].lower()
        keyword, keyword_index_of_vocabulary, new_candidate_keywords_list, candidate, candidate_text_rank = \
            get_index_of_vocabulary(vocabulary, keyword, candidate_keywords_list)
        if candidate:
            # 如果启用了候选词,则更新候选词列表
            keyword_text_rank = candidate_text_rank
        else:
            keyword_text_rank = tuples[1]

        candidate_keywords_list = new_candidate_keywords_list
        # 根据索引，找到关键词对应tf_idf的值
        keyword_tf_idf = tf_idf[0][keyword_index_of_vocabulary]
        # 获得单词的值
        keyword_value = (keyword_tf_idf + keyword_text_rank) / 2
        text_vector.append(keyword_value)

    if len(text_vector) < 20:
        text_vector += (16 - len(text_vector)) * [0]
    return text_vector


def get_sorted_values(row_content):
    sorted_values_container = []  # 初始化为空list
    keys = row_content.keys()
    for key in keys:
        sorted_values_container.append(row_content[key])
    return sorted_values_container


if __name__ == '__main__':
    """加载模型"""
    # 结巴加载用户自定义词典
    jieba.load_userdict(r"./baidu_voca.txt")
    # 结巴加载停用词典
    stop_words_list = [i.strip() for i in codecs.open("./stopwords/baidu_stopwords.txt", encoding='utf-8').readlines()]
    # 加载词cv词汇表
    cv_feature_path = './data_word2vec/feature.pkl'
    loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(cv_feature_path, "rb")))
    cv_vocabulary = loaded_vec.vocabulary
    print(cv_vocabulary)
    # 加载cv_tf_idf模型
    tf_idf_transformer_path = './data_word2vec/tf_idf_Transformer.pkl'
    tf_idf_transformer = pickle.load(open(tf_idf_transformer_path, "rb"))
    # 加载写入的csv文件
    output_file = open("./data_word2vec/result.csv", 'w', newline='', encoding='utf-8')
    writer = csv.writer(output_file)
    csv_head = {"Column1": "index", "Column2": "text_vector"}
    sorted_values = get_sorted_values(csv_head)
    writer.writerow(sorted_values)
    # 加载数据,利用生成器产生一条交互信息
    all_interaction_records_content = load_data()
    '''
    交互记录interaction_record_content;
        第一列：index;
        第二列：text;
        第三列：start_time;
        第四列：end_time
    '''
    for interaction_record_content in all_interaction_records_content:
        #  获取每条交互信息的text
        if interaction_record_content[0] == 0:
            continue
        print("正在处理第"+str(interaction_record_content[0])+"条交互信息的内容")
        if not interaction_record_content[1]:
            print("该条交互信息没有内容")
            print("----------------------------------------------------------------")
            continue
        else:
            print("交互信息的内容为: "+interaction_record_content[1])
        interaction_record_content_text = interaction_record_content[1]

        # 分词处理
        interaction_record_content_text_word_segments = cut_word(interaction_record_content_text, stop_words_list)
        print("内容分词结果为: "+interaction_record_content_text_word_segments)

        # 获取TextRank
        interaction_record_content_text_word_segments_text_rank = \
            get_text_rank(interaction_record_content_text_word_segments, stop_words_list)
        print("内容分词的TextRank结果为: " + str(interaction_record_content_text_word_segments_text_rank))

        # 获取Tf_Idf
        interaction_record_content_text_word_segments_tf_idf = \
            get_cv_tf_idf(interaction_record_content_text_word_segments, loaded_vec, tf_idf_transformer)
        print("内容分词的Tf_Idf结果为: " + str(interaction_record_content_text_word_segments_tf_idf))

        # 获取句子向量
        interaction_record_content_text_vector = get_text_vector(interaction_record_content_text_word_segments_text_rank, interaction_record_content_text_word_segments_tf_idf, cv_vocabulary)
        print("最终内容的向量化结果为: "+str(interaction_record_content_text_vector))

        # 句子向量写入文件
        print("写入csv....")
        row = {"Column1": interaction_record_content[0], "Column2": interaction_record_content_text_vector}
        sorted_values = get_sorted_values(row)
        writer.writerow(sorted_values)
        print("----------------------------------------------------------------")\


    output_file.close()

