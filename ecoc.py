import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from sympy.combinatorics.graycode import GrayCode
from scipy.spatial.distance import hamming

from copy import deepcopy


class EcocEstimator(BaseEstimator, ClassifierMixin):
    """
    ECOC meta classifier (works without sklearn.OutputCodeClassifier restrictions)
    http://www.cs.cmu.edu/~aberger/pdf/ecoc.pdf
    """

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def __init__(self, base_estimator, code_size=1.0, random_state=42):
        self.random_state = random_state
        self.base_estimator = base_estimator
        self.code_size = code_size
        self.error_code = None
        self.estimators = None
        self.class2id = None
        self.id2class = None

    def _make_mappings(self, unique_classes):
        class2id = {v: k for k, v in enumerate(unique_classes)}
        id2class = {k: v for k, v in enumerate(unique_classes)}

        return class2id, id2class

    def _map_values(self, classes, mapping):
        return np.array(list(
            map(lambda x: mapping[x], classes)
        ))

    def _generate_gray_codes(self, code_size):
        gray = np.array([[int(j) for j in i] for i in GrayCode(code_size).generate_gray()])
        total_values_sum = gray.sum(axis=1)
        to_delete = [total_values_sum.argmax(), total_values_sum.argmin()]
        return np.delete(gray, to_delete, axis=0)

    def _hamming_corrmat(self, gray):
        result = []

        # TODO: reimplement a method, very slow and unoptimized
        for i in gray:
            row = []
            for j in gray:
                row.append(hamming(i, j))
            result.append(row)

        return np.array(result)

    def _greedy_search(self, a, size=None):
        if size is None:
            size = int(np.log2(a.shape[0] + 1))

        result = []
        current = np.random.randint(0, a.shape[0])
        result.append(current)

        for i in range(size - 1):
            row_sum = a[:, result].sum(1)
            min_border = np.percentile(row_sum, 85)
            valid_vals = row_sum[row_sum >= min_border]
            max_index = np.random.choice(
                np.argwhere(np.isin(row_sum, valid_vals)).flatten()
                , 1).squeeze()

            result.append(int(max_index))
            a[current, max_index] = -1
            a[max_index, current] = -1
            current = max_index
        return result

    def fit(self, X, y=None):
        if y is None:
            raise Exception('y should be initialized')

        unique_classes = np.unique(y)
        self.class2id, self.id2class = self._make_mappings(unique_classes)
        _y = self._map_values(y, self.class2id)

        classes_size = unique_classes.shape[0]
        code_size = int(classes_size * self.code_size)

        if code_size < np.log2(classes_size):
            print('inapropriate code size, try to adjust it')

        if code_size > 8:
            # if code size is too small random initialization
            # is not robust enough, it is more rational to use
            # gray codes to generate codes with maximum distance

            np.random.seed(self.random_state)
            self.error_code = np.random.randint(2, size=(classes_size, code_size))
            while set([0, classes_size]).intersection(self.error_code.sum(0)) \
                    or set([0, code_size]).intersection(self.error_code.sum(1)):
                self.error_code = np.random.randint(2, size=(classes_size, code_size))
        else:
            gray_codes = self._generate_gray_codes(classes_size)
            gray_corrmat = self._hamming_corrmat(gray_codes)
            codes_id = self._greedy_search(gray_corrmat, code_size)
            self.error_code = np.array(gray_codes[codes_id]).T

        # iterate over error codez
        estimators = []
        for i in range(self.error_code.shape[1]):
            # print('fitting %d-th classifier out of %d' % (i + 1, code_size))

            classifier_mapping = self.error_code[:, i]
            # print(classifier_mapping)

            positive = np.argwhere(classifier_mapping == 1).flatten()
            negative = np.argwhere(classifier_mapping == 0).flatten()

            _y_i = _y.copy()

            _y_i[np.isin(_y_i, positive)] = -2
            _y_i[np.isin(_y_i, negative)] = -1
            _y_i[_y_i == -2] = 1

            estimator = clone(self.base_estimator)
            estimator.fit(X, _y_i)

            estimators.append(estimator)

        self.estimators = estimators

    def predict(self, X, y=None):
        """
        Extended ECOC model intefrace, implementation is based on
        http://www.cs.cmu.edu/~aberger/pdf/ecoc.pdf
        :param X: classification input (can be a sparse matrix)
        :return: numpy array of classification results
        """

        if not hasattr(X, 'shape'):
            raise Exception('X must be a numpy object')

        predicted = []
        for i in range(len(self.estimators)):
            predicted.append(self.estimators[i].predict(X) * -1)

        predicted = np.array(predicted).T

        scores = []
        for i in range(predicted.shape[0]):
            tiled_predictions = np.tile(predicted[i], (self.error_code.shape[0], 1))

            class_score = np.sum(np.abs(self.error_code - tiled_predictions), axis=1)
            class_score = np.argmax(class_score)

            scores.append(class_score)

        return self._map_values(np.array(scores), self.id2class)

    def predict_proba(self, X, y=None):
        """
        Extended ECOC model intefrace, implementation is based on
        http://www.cs.cmu.edu/~aberger/pdf/ecoc.pdf
        :param X: classification input (can be a sparse matrix)
        :return: numpy array of classification results
        """
        if not hasattr(X, 'shape'):
            raise Exception('X must be a numpy object')

        predicted = []
        for i in range(len(self.estimators)):
            predicted.append(self.estimators[i].predict(X) * -1)

        predicted = np.array(predicted).T

        scores = []
        for i in range(predicted.shape[0]):
            tiled_predictions = np.tile(predicted[i], (self.error_code.shape[0], 1))

            class_score = np.sum(np.abs(self.error_code - tiled_predictions), axis=1)
            class_score = EcocEstimator.softmax(class_score)

            scores.append(class_score)

        return np.array(scores)

    def predict_top_n(self, X, n=5):
        """
        Predict top N most relevant classes for each record
        :param X: classification input (can be a sparse matrix)
        :param n: how much top results should be given
        :return: numpy matrix of top predictions
        """
        result = self.predict_proba(X)
        sorted_probas = deepcopy(result)
        sorted_probas.sort(axis=1)
        sorted_probas = sorted_probas[:, -n:][:, ::-1]
        class_mapper = np.vectorize(lambda x: self.id2class[x])

        classes = class_mapper(result.argsort(axis=1)[:, -n:][:, ::-1])

        sorted_probas = [[float(j) for j in i] for i in sorted_probas.tolist()]
        classes = [[j for j in i] for i in classes.tolist()]

        return list(map(lambda x: dict(zip(x[0], x[1])), zip(classes, sorted_probas)))

    def shrink_prediction(self, X, y, n=5):
        """
        Reduce prediction of predict_top_n with the following rule:
        if the true result is in top, then replace the prediction with true result
        otherwise replace prediction with the most significant prediction
        :param X: classification input (can be a sparse matrix)
        :param y: real labels
        :param n: how much top results should be given
        :return: numpy array of shrinked predictions
        """
        top_args = self.predict_top_n(X, n=n)
        shrinked = []
        for i, r in zip(top_args, y):
            if r in i:
                shrinked.append(r)
            else:
                keys, vals = list(zip(*i.items()))
                shrinked.append(keys[np.argmax(vals)])

        return shrinked
