import cv2
import numpy as np


class ScoreQueue:
    def __init__(self, max_length=3):
        self._max_length = max_length
        self.clear()

    def __repr__(self):
        return repr(self._queue)

    def __getitem__(self, index):
        return self._queue[index]

    def clear(self):
        self._queue = [''] * self._max_length

    @property
    def perfect_duplicates(self):
        if self._queue.count(self._queue[0]) == len(self._queue) and \
           self._queue[0] != '' and self._queue[0]:
            return True
        return False

    def push(self, new_element):
        self._queue = self._queue[1:] + [new_element]


class ScoreParser:
    def __init__(self):
        self._current_score = '0-0'
        self.queue = ScoreQueue(max_length=3)
        self._serving = True
        self._templates = {
            '0-0': cv2.imread('/src/saved_images/0_0.jpg', -1),
            '0-15': cv2.imread('/src/saved_images/0_15.jpg', -1),
            '0-30': cv2.imread('/src/saved_images/0_30.jpg', -1),
            '0-40': cv2.imread('/src/saved_images/0_40.jpg', -1),
            '15-0': cv2.imread('/src/saved_images/15_0.jpg', -1),
            '15-15': cv2.imread('/src/saved_images/15_15.jpg', -1),
            '15-30': cv2.imread('/src/saved_images/15_30.jpg', -1),
            '15-40': cv2.imread('/src/saved_images/15_40.jpg', -1),
            '30-0': cv2.imread('/src/saved_images/30_0.jpg', -1),
            '30-15': cv2.imread('/src/saved_images/30_15.jpg', -1),
            '30-30': cv2.imread('/src/saved_images/30_30.jpg', -1),
            '30-40': cv2.imread('/src/saved_images/30_40.jpg', -1),
            '40-0': cv2.imread('/src/saved_images/40_0.jpg', -1),
            '40-15': cv2.imread('/src/saved_images/40_15.jpg', -1),
            '40-30': cv2.imread('/src/saved_images/40_30.jpg', -1),
            'deuce': cv2.imread('/src/saved_images/deuce.jpg', -1),
            'adv-40': cv2.imread('/src/saved_images/adv_40.jpg', -1),
            #'40-adv': cv2.imread('/src/saved_images/40_adv.jpg', -1),
            'match-complete': cv2.imread('/src/saved_images/match_complete.jpg', -1)
        }

    @property
    def current_score(self):
        return self._current_score

    def _parse_image(self, pixel_array):
        img = cv2.cvtColor(pixel_array[165:275, 145:495, :], cv2.COLOR_BGR2HSV)
        lower = np.array([85, 200, 220])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(img, lower, upper)
        img = cv2.bitwise_and(img, img, mask=mask)
        lower = np.array([5, 5, 5])
        upper = np.array([255, 255, 255])
        img[mask != 0] = [255, 255, 255]
        return img

    def _best_fit_score(self, img):
        best = 0.0
        best_key = None

        for key, image in self._templates.items():
            intersection = np.sum(np.logical_and(img, image))
            union = np.sum(np.logical_or(img, image))
            miou = intersection / float(union)
            if miou > best and miou > 0.35:
                best = miou
                best_key = key
        self.queue.push(best_key)
        return best_key

    def _determine_deuce(self, new_score, old_score):
        if new_score == 'deuce':
            old_server, old_returner = old_score.split('-')
            # The server lost the point, and the agent is serving.
            if old_server == 'adv' and self._serving:
                return -1.0
            # The server won the point, and the agent is serving.
            if old_returner == 'adv' and self._serving:
                return 1.0
            # The server won the point, and the agent is serving.
            if old_server == '30' and self._serving:
                return 1.0
            # The server lost the point, and the agent is serving.
            if old_returner == '30' and self._serving:
                return -1.0
            # The server lost the point, and the agent is not serving.
            if old_server == 'adv' and not self._serving:
                return 1.0
            # The server won the point, and the agent is not serving.
            if old_returner == 'adv' and not self._serving:
                return -1.0
            # The server won the point, and the agent is not serving.
            if old_server == '30' and not self._serving:
                return -1.0
            # The server lost the point, and the agent is not serving.
            if old_returner == '30' and not self._serving:
                return 1.0
        new_server, new_returner = new_score.split('-')
        # The server won the point, and the agent is serving.
        if new_server == 'adv' and self._serving:
            return 1.0
        # The server lost the point, and the agent is serving.
        if new_returner == 'adv' and self._serving:
            return -1.0
        # The server won the point, and the agent is not serving.
        if new_server == 'adv' and not self._serving:
            return -1.0
        # The server lost the point, and the agent is not serving.
        if new_returner == 'adv' and not self._serving:
            return 1.0

    def _determine_match_winner(self, old_score):
        old_server, old_returner = old_score.split('-')

        # The server won the point, and the agent is serving.
        if old_server in ['6', 'adv'] and self._serving:
            return 1.0
        # The returner won the point, and the agent is serving.
        if old_returner in ['6', 'adv'] and self._serving:
            return -1.0
        # The server won the point, and the agent is not serving.
        if old_server in ['6', 'adv'] and not self._serving:
            return -1.0
        # The returner won the point, and the agent is not serving.
        if old_returner in ['6', 'adv'] and not self._serving:
            return 1.0

    def _determine_winner(self, new_score, old_score):
        if new_score == 'deuce' or 'adv' in new_score:
            return self._determine_deuce(new_score, old_score)
        if 'adv' in old_score:
            old_score = '0-0'
        if new_score == 'match-complete':
            return self._determine_match_winner(new_score)

        old_server, old_returner = old_score.split('-')
        new_server, new_returner = new_score.split('-')

        # The server won the point, and the agent is serving.
        if int(new_server) > int(old_server) and self._serving:
            return 1.0
        # The returner won the point, and the agent is serving.
        if int(new_returner) > int(old_returner) and self._serving:
            return -1.0
        # The server won the point, and the agent is returning.
        if int(new_server) > int(old_server) and not self._serving:
            return -1.0
        # The returner won the point, and the agent is returning.
        if int(new_returner) > int(old_returner) and not self._serving:
            return 1.0
        return 0.0

    def reward(self, pixel_array):
        reward = 0.0
        img = self._parse_image(pixel_array)
        best_fit = self._best_fit_score(img)
        if self.queue.perfect_duplicates and best_fit != self.current_score:
            reward = self._determine_winner(best_fit, self.current_score)
            self.current_score = best_fit
        return reward

    @property
    def match_complete(self):
        # Return True only if the "Game, Set, Match" image is found on-screen.
        if self.queue.perfect_duplicates and self.queue[0] == 'match-complete':
            print('Match Complete')
            self.queue.clear()
            return True
        return False
