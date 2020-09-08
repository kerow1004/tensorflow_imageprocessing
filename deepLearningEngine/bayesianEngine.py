import math, sys, pickle, os
from konlpy.tag import Twitter
from konlpy.tag import Okt
import numpy
import random

def dataListDir(bookId):
    if not (os.path.isdir('../text_bin_file/'+str(bookId))):
        os.makedirs(os.path.join('../text_bin_file/'+str(bookId)))
        listDir = ['../text_bin_file/' + str(bookId) + '/words_set.bin',
                   '../text_bin_file/' + str(bookId) + '/word_dict.bin',
                   '../text_bin_file/' + str(bookId) + '/category_dict.bin']
        return listDir
    elif len(os.listdir('../text_bin_file/'+str(bookId))) != 3:
        listDir = ['../text_bin_file/'+str(bookId)+'/words_set.bin',
                   '../text_bin_file/'+str(bookId)+'/word_dict.bin',
                   '../text_bin_file/'+str(bookId)+'/category_dict.bin']
        return listDir
    else:
        listDir = os.listdir('./text_bin_file/'+str(bookId))
        return listDir

class BayesianFilter:
    """ 베이지안 필터 """

    def __init__(self):
        self.words = set()  # 출현한 단어 기록
        self.word_dict = {}  # 카테고리마다의 출현 횟수 기록
        self.category_dict = {}  # 카테고리 출현 횟수 기록

    # 형태소 분석하기
    def split(self, text):
        results = []
        # twitter = Twitter()
        twitter = Okt()
        # 단어의 기본형 사용
        malist = twitter.pos(text, norm=True, stem=True)
        for word in malist:
            # 어미/조사/구두점 등은 대상에서 제외
            if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                results.append(word[0])
        return results

    # 단어와 카테고리의 출현 횟수 세기
    def inc_word(self, word, category):
        # 단어를 카테고리에 추가하기
        if not category in self.word_dict:
            self.word_dict[category] = {}
        if not word in self.word_dict[category]:
            self.word_dict[category][word] = 0
        self.word_dict[category][word] += 1
        self.words.add(word)

    def inc_category(self, category):
        # 카테고리 계산하기
        if not category in self.category_dict:
            self.category_dict[category] = 0
        self.category_dict[category] += 1

    # 텍스트 학습하기
    def fit(self, text, category):
        """ 텍스트 학습 """
        word_list = self.split(text)
        for word in word_list:
            self.inc_word(word, category)
        self.inc_category(category)

    # 단어 리스트에 점수 매기기
    def score(self, words, category):
        score = math.log(self.category_prob(category))
        for word in words:
            score += math.log(self.word_prob(word, category))
        return score

    # 예측하기
    def predict(self, text):
        best_category = None
        max_score = -sys.maxsize
        words = self.split(text)
        score_list = []
        alike_score = []
        numpy_list = []
        for category in self.category_dict.keys():
            try:
                score = self.score(words, category)
            except KeyError:
                pass
            score_list.append((score, category))
            if score > max_score:
                max_score = score
                best_category = category

        prob_random = random.sample(score_list, 10)
        sample_value = random.sample(score_list, 1)

        for i in range(10):
            numpy_list.append(prob_random[i][0])
        np_mean = numpy.mean(numpy_list)

        if np_mean == sample_value[0][0]:
            return best_category, score_list, alike_score
        else:
            score_list.sort(key=lambda element:element[0], reverse=True)

            for i in range(5):
                alike_score.append(score_list[i][1])
            return best_category, score_list, alike_score

    # 카테고리 내부의 단어 출현 횟수 구하기
    def get_word_count(self, word, category):
        if word in self.word_dict[category]:
            return self.word_dict[category][word]
        else:
            return 0

    # 카테고리 계산
    def category_prob(self, category):
        sum_categories = sum(self.category_dict.values())
        category_v = self.category_dict[category]
        return category_v / sum_categories

    # 카테고리 내부의 단어 출현 비율 계산
    def word_prob(self, word, category):
        n = self.get_word_count(word, category) + 1
        d = sum(self.word_dict[category].values()) + len(self.words)
        return n / d

    # 단어분류 저장
    def word_save(self, bookId):
        listDir = dataListDir(bookId)
        for file_bin in listDir:
            with open(file_bin, 'wb') as file:
                if os.path.splitext(file_bin)[0] == 'words_set':
                    pickle.dump(self.words, file)
                elif os.path.splitext(file_bin)[0] == 'word_dict':
                    pickle.dump(self.word_dict, file)
                else:
                    pickle.dump(self.category_dict, file)
    # 단어분류 로드
    def word_load(self, bookId):
        listDir = dataListDir(bookId)
        for file_bin in listDir:
            with open(file_bin, 'rb') as file:
                while True:
                    try:
                        data = pickle.load(file)
                    except EOFError:
                        break
                if os.path.splitext(file_bin)[0] == 'words_set':
                    self.words = data
                elif os.path.splitext(file_bin)[0] == 'word_dict':
                    self.word_dict = data
                else:
                    self.category_dict = data