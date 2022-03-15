-- References

https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/trees.pdf

https://www.youtube.com/watch?v=D0efHEJsfHo&t=776s

http://mlwiki.org/index.php/Cost-Complexity_Pruning

-- k-fold cross validation

    Returns a model trained on _all_ of the training data

    pass all data, and iteratively build trees with k-1/k of the
    training data. Then for each value of k, determine the average
    Tree Score across all data splits.

-- pruning

    bottom-up
    * reduced-error pruning
    * minimum cost complexity pruning

        Tree Score = SSR + alpha * |T|  (num leaves)
            (higher the alpha, higher the penalty for having leaves)

        Note this is also dependent on train/test data split, which we
        average using k-fold cross validation.

    * minimum error pruning

    top-down
    * pessimistic error pruning

# verification of regression tree using boston housing dataset

The alphas returned from the cost-complexity pruning algorithm correspond to trees that have the following average residuals across the _training_ dataset.

INFO[0001] alpha 0, avg residual 14.134395421607389     
INFO[0001] alpha 0.12883399209486168, avg residual 14.148010046113317 
INFO[0001] alpha 0.1653326745718049, avg residual 14.185322598514801 
INFO[0001] alpha 0.16639262187088283, avg residual 14.294436412237312 
INFO[0001] alpha 0.1885079051383401, avg residual 14.302006480508961 
INFO[0001] alpha 0.31038537549407086, avg residual 14.315709543750067 
INFO[0001] alpha 0.39030303030303015, avg residual 14.384718627724828 
INFO[0001] alpha 0.39562648221343866, avg residual 14.581975874101639 
INFO[0001] alpha 0.42822134387351773, avg residual 14.60797850914775 
INFO[0001] alpha 0.4476366145931363, avg residual 14.632440697020266 
INFO[0001] alpha 0.46552795031055916, avg residual 14.653141893057436 
INFO[0001] alpha 0.4960061484409311, avg residual 17.111619118730296 
INFO[0001] alpha 0.7196160361377755, avg residual 17.184331428159997 
INFO[0001] alpha 0.7527879728966684, avg residual 17.21873995441635 
INFO[0001] alpha 0.7842687747035573, avg residual 17.32469313167387 
INFO[0001] alpha 0.8549160079051387, avg residual 17.50569044957337 
INFO[0001] alpha 0.9169193487671752, avg residual 17.537741043906575 
INFO[0001] alpha 1.043017127799736, avg residual 17.553606217380448 
INFO[0001] alpha 1.0943305335968383, avg residual 17.561193562100943 
INFO[0001] alpha 1.1756884057971015, avg residual 17.575857923102262 
INFO[0001] alpha 1.3147481351600332, avg residual 17.642782325930554 
INFO[0001] alpha 1.363734797810884, avg residual 17.64885070448294 
INFO[0001] alpha 1.4126577733860342, avg residual 17.739158623489942 
INFO[0001] alpha 1.4584085841694536, avg residual 17.841685839631293 
INFO[0001] alpha 1.5298256815648132, avg residual 17.899212635673834 
INFO[0001] alpha 1.698695652173913, avg residual 17.91000578455394 
INFO[0001] alpha 1.7142770092226611, avg residual 18.768780158730486 
INFO[0001] alpha 2.2946197149359207, avg residual 18.820889252457803 
INFO[0001] alpha 2.669404933896688, avg residual 18.93028713105861 
INFO[0001] alpha 2.6697289695372706, avg residual 19.81360930537091 
INFO[0001] alpha 3.467875494071145, avg residual 19.87558924974215 
INFO[0001] alpha 3.5513955609607777, avg residual 19.948430538887788 
INFO[0001] alpha 3.8417485413137573, avg residual 19.970355118996512 
INFO[0001] alpha 4.013348390739696, avg residual 20.365796253948517 
INFO[0001] alpha 4.055975942371199, avg residual 20.65320518951057 
INFO[0001] alpha 4.175004919952169, avg residual 21.626034571197373 
INFO[0001] alpha 4.3688932806324114, avg residual 21.801771877369458 
INFO[0001] alpha 4.861062582345191, avg residual 22.49773465734311 
INFO[0001] alpha 9.092344678670765, avg residual 23.682930286402573 
INFO[0001] alpha 10.863199110671939, avg residual 24.18075728830566 
INFO[0001] alpha 11.144254281949934, avg residual 26.84874003382101 
INFO[0001] alpha 11.359106424658487, avg residual 26.86842014022945 
INFO[0001] alpha 13.293537256624216, avg residual 27.229961657615355 
INFO[0001] alpha 16.848179331888318, avg residual 27.357791179357594 
INFO[0001] alpha 64.57426297958334, avg residual 28.294278731131197 
INFO[0001] alpha 133.2095039943955, avg residual 29.437580130043788 
INFO[0001] alpha 167.29907631644315, avg residual 29.937975886431595 
INFO[0001] alpha 439.84432703667676, avg residual 32.92121308837762 
INFO[0001] alpha 621.0306084614934, avg residual 36.3987567628702 
INFO[0001] alpha 739.3793217781949, avg residual 43.51964252487374 
INFO[0001] alpha 837.3864710344569, avg residual 61.267903297185946 
INFO[0001] alpha 1026.5185269754365, avg residual 84.4195561561656

When running cross validation it looks like average residual is pretty different across folds, which in turn gives very different values for alpha. Will need to investigate this further. Here are some typical results from 10-fold cross validation:

INFO[0002] * fold 0: alpha 0, error: 911.9933467781559  
INFO[0003] * fold 1: alpha 15.414289473684207, error: 599.9848234221355 
INFO[0004] * fold 2: alpha 7.063954870729459, error: 1168.1684906780179 
INFO[0004] * fold 3: alpha 0.29686925647451967, error: 2082.105952653071 
INFO[0005] * fold 4: alpha 15.08986257309941, error: 2491.851239853936 
INFO[0005] * fold 5: alpha 1.2758108552631582, error: 1543.555675071225 
INFO[0006] * fold 6: alpha 651.0241925560358, error: 2373.5789896637143 
INFO[0006] * fold 7: alpha 2.3170317286271227, error: 1363.5892092966592 
INFO[0007] * fold 8: alpha 1.7122904483430803, error: 1209.2366845834206 
INFO[0007] * fold 9: alpha 59.18484461152882, error: 1927.6639759480731

INFO[0001] * fold 0: alpha 7.3873261278195494, error: 873.6934505697388 
INFO[0001] * fold 1: alpha 1.3629385964912273, error: 1118.542076681813 
INFO[0002] * fold 2: alpha 11.505924077434969, error: 1967.6345849048907 
INFO[0002] * fold 3: alpha 0.6515423976608188, error: 1863.768152901023 
INFO[0003] * fold 4: alpha 6.725902255639101, error: 1024.840594792638 
INFO[0003] * fold 5: alpha 0.10333333333333328, error: 2015.6219700947977 
INFO[0004] * fold 6: alpha 223.06815241228057, error: 1083.4867525770953 
INFO[0004] * fold 7: alpha 6.710175438596491, error: 2095.471093103781 
INFO[0005] * fold 8: alpha 136.45015125533132, error: 793.516878680348 
INFO[0005] * fold 9: alpha 18.12495875104426, error: 2147.9405265122323

INFO[0001] * fold 0: alpha 4.805873538011697, error: 1898.6673765948942 
INFO[0001] * fold 1: alpha 4.680736336032389, error: 1480.5795186811247 
INFO[0002] * fold 2: alpha 1.369949874686717, error: 837.2681606262881 
INFO[0002] * fold 3: alpha 209.5403155941295, error: 3682.7290853603095 
INFO[0003] * fold 4: alpha 1.405701754385965, error: 1290.252069084311 
INFO[0003] * fold 5: alpha 3.3733304093567273, error: 1217.268751528195 
INFO[0004] * fold 6: alpha 17.46343440122044, error: 1063.6356448669746 
INFO[0004] * fold 7: alpha 32.8669200779727, error: 2693.1774483892314 
INFO[0005] * fold 8: alpha 4.16376096491228, error: 809.8810069806822 
INFO[0005] * fold 9: alpha 1.0796900584795321, error: 534.8713367940638