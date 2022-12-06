import numpy as np
import math
import random as rnd


class KNNClassifier:

    def pearson(self, mat, u_t, i_t):
        if i_t == -1:  # when you don't need an item to compare
            new_mat = mat
            new_ut = u_t
        else:
            new_mat = np.delete(mat, i_t, 1)  # delete the item column for the calculations
            new_ut = np.delete(u_t, i_t, 0)

        similarity = []
        for i in range(new_mat.shape[0]):

            multi = new_mat[i]*new_ut  # zeroing the non-overlapping elements
            nonzero_idx = np.where(multi > 0)[0].tolist()  # get the idex location of non-zero elements
            if len(nonzero_idx) != 0:  # if non-zero elements are exist
                mean_mat = sum(new_mat[i][nonzero_idx])/len(nonzero_idx)
                mean_u = sum(new_ut[nonzero_idx])/len(nonzero_idx)

                cov = 0
                vart = 0
                varu = 0
                for j in nonzero_idx:
                    # if new_mat[i][j] > 0 and new_ut[j] > 0:
                    cov += ((new_mat[i][j]-mean_mat)*(new_ut[j]-mean_u))
                    vart += ((new_mat[i][j]-mean_mat)*(new_mat[i][j]-mean_mat))
                    varu += ((new_ut[j]-mean_u)*(new_ut[j]-mean_u))
                if vart == 0 or varu == 0:  # avoid division by zero. Because we consider only the overlapping items
                    sim = 0
                else:
                    sim = cov/(math.sqrt(vart*varu))
            else:
                sim = 0
            similarity.append(sim)
        return similarity

    def K_neibghorsPred(self, mat, K, u_t, i_t):
        itemF = True  # when there is an item to predict
        similarity = self.pearson(mat, u_t, i_t)  # compute the Pearson correlation coefficient
        nsim = np.array(similarity)
        idx = np.argsort(nsim)  # sort the similarities in ascending order
        idx = idx[::-1]  # sort the indices in descending order
        if i_t == -1:  # if there is no item to predict
            itemF = False
        if itemF: ratings = mat[idx[:K], i_t]  # get the ratings given by the highest K users for the given item
        sims = nsim[idx[:K]]  # extract the highest K similarities

        deno = 0
        neu = 0
        for i in range(sims.shape[0]):
            if itemF:
                if ratings[i] > 0 and sims[i] >= 0:  # filter the positive similarities and existing ratings
                    neu += (ratings[i]*sims[i])
                    deno += sims[i]
        if deno == 0:  # avoid division by zero. since we only consider the overlapped rated-items.
            pred_rating = sum(mat[:][i_t])/mat.shape[0]  # if there is no neighbor for rate prediction
        else:
            pred_rating = neu/deno
        return pred_rating, idx[:K], sims

    def recommendations(self, mat, u_t, books, N, K):
        '''N - number of recommendations
           K - Number of neibghors of the system '''
        zidx = np.where(u_t == 0)[0].tolist()  # get the non-rated item indices
        booksList = books[zidx]  # get the related book names
        pred_ratings = []
        if len(zidx) != 0:
            for i in zidx:
                prediction, neighbor, sims = self.K_neibghorsPred(mat, K, u_t, i)
                pred_ratings.append(prediction)
        npred_ratings = np.array(pred_ratings)
        pidx = np.argsort(npred_ratings)  # sort predicted ratings ascending order
        pidx = pidx[::-1]  # sort the indices in descending order
        if len(zidx) < N:  # if number of recommendations are larger than the total items received.
            rec_books = booksList[pidx]
            ratings = npred_ratings[pidx]
        else:
            rec_books = booksList[pidx[:N]]
            ratings = npred_ratings[pidx[:N]]

        return rec_books, ratings

    def evaluation(self, mat, K, ts_ut):
        pred_error = 0
        users = ts_ut.shape[0]  # number users in the training data
        cases = 0  # count the predictions
        for i in range(users):
            u_t = ts_ut[i]
            ridx = np.where(u_t > 0)[0].tolist()  # select the indices where non-zeros values are present
            for i_t in ridx:
                pred_rating, neighbour, sims = self.K_neibghorsPred(mat, K, u_t, i_t)
                pred_error += abs(pred_rating - u_t[i_t])
                cases += 1
        mae = pred_error/cases
        return mae

    def eval_joker(self, mat, K, ts_ut):
        users = ts_ut.shape[0]  # number users in the training data
        pred_error_list = []
        for i in range(users):
            non_zeroidx = np.where(ts_ut[i] > 0)[0]  # get the rated item indices
            rindx = rnd.sample(range(len(non_zeroidx)), len(non_zeroidx))  # randomize the indices
            len20 = math.floor(len(rindx) * 0.2)  # select first 20% of the indices
            ets_ut = ts_ut[i]  # make a copy of current user profile

            # set the current ratings as 0 for 20% indices and then other 80% only be used for predic
            ets_ut[non_zeroidx[rindx[:len20]]] = 0
            u_t = ets_ut
            ridx = np.where(u_t == 0)[0].tolist()  # select the indices where zeros values are present
            user_pred_error = 0
            cases = 0  # count the predictions
            for i_t in ridx:
                pred_rating, neighbour, sims = self.K_neibghorsPred(mat, K, u_t, i_t)
                user_pred_error += abs(pred_rating - ts_ut[i][i_t])
                cases += 1
            pred_error_list.append(user_pred_error / cases)
            print(i, " - ", pred_error_list[i])

        return pred_error_list, sum(pred_error_list)/len(pred_error_list)
