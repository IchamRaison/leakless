# D3 — observation intermédiaire au pas 100

Copie des observations déjà produites par l'unique fit préinscrit `42920f6d…`, PID100634, handle20214. Ce dossier n'est ni un run terminé ni un checkpoint livrable : le fit continue, aucun reload n'a encore eu lieu. Aucun fichier dans le dossier distant scellé n'a été ajouté pour cette revue.

Point0 : NLL binaire3,244997234820116 et16/32décisions correctes, identiques à D0-A initial. Point100 : NLL0,5749842273444788,24/32corrects (12TP,12TN,4FP,4FN), AUC clip/groupe0,76953125. Critère préinscrit non atteint : il exige32/32corrects **et** NLL<0,1. Ces32clips sont vus pendant l'apprentissage ; aucune amélioration hors groupes/terrain n'est démontrée.

`verification.log` : les32IDs/ordre/cibles/groupes ont été comparés à la préinscription. Sommes des contributions des tokens de chaque classe revérifiées ; softmax recalculé depuis les deux sommes ; NLL recomposée par `max(d,0)+log1p(exp(-abs(d)))`, où `d=logp_autre−logp_vraie`. Probabilités et NLL concordent à1e-14. Tous les agrégats sont reproduits par `diagnose_causal.partition_summary`, sans nouvelle inférence. La revue indépendante du préfixe train40pas/10époques/320présentations a aussi vérifié l'ordre, les cibles, les comptes, le clipping et AdamW ; son périmètre n'est pas celui d'un audit final du fit.
