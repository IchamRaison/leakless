# Entire - installation et vérification

Icham

## Vérifié sur le poste Linux d'Icham

- Entire CLI mis à jour de 0.7.7 à 0.10.6, installateur officiel stable et checksum de release vérifié.
- 12 skills de https://github.com/entireio/skills installés dans `.agents/skills/` du dépôt code ; noms et frontmatters validés, skills-lock.json cohérent, liens d'agents résolus.
- Commande demandée exécutée : `npx --yes skills add https://github.com/entireio/skills --all`.
- Le skill search était rejeté par l'installateur à cause d'une description YAML non échappée. Description mise entre guillemets JSON valides YAML, puis skill installé. Réparation locale et provenance documentées dans `.agents/ENTIRE-INSTALL.md`. Fichier corrigé committé ; source temporaire du lock à ne pas utiliser comme source distante pour une réinstallation.
- Hermes profil default configuré pour lire ce répertoire via skills.external_dirs ; les 12 apparaissent dans hermes skills list. Aucun autre profil modifié. La découverte de skills n'est pas une preuve de capture automatique de sessions Hermes.
- Hooks Git et configurations Claude Code/OpenCode/Pi vérifiés après rafraîchissement des intégrations. Huit intégrations configurées, sans prétendre toutes les avoir exécutées.
- Capture Codex réellement constatée : session `01a095b4-5388-7893-b9a2-0f58a30ae6de`, trois checkpoints listés. Lecture locale du checkpoint `0664fd3bb2f7` réussie, lié au commit `a5049d1`. Ce sont de vraies traces existantes, pas une session de test inventée.
- Backend historique de checkpoints conservé (git-branch), télémétrie désactivée et push automatique des sessions toujours désactivé.
- Skills et hooks poussés au dépôt code, commit `916986f`, lecture distante confirmée.

## Restant à faire par l'utilisateur

1. Lancer `entire login` et terminer le parcours d'authentification dans le navigateur. Le CLI indique Not logged in ; existence du compte web non établie. Aucun mot de passe à communiquer à l'agent.
2. Dans Codex, ouvrir `/hooks` pour approuver les trois nouveaux hooks ajoutés par cette version : session_end, subagent_start, subagent_stop. Les approbations existantes sont présentes, ces trois nouvelles approbations manquent. Ne pas fabriquer les trusted_hash à la main pour contourner la revue.
3. Si l'équipe veut utiliser les sessions sur entire.io et la recherche indexée : décider explicitement de publier les checkpoints après vérification de leur contenu et des permissions du repo. Aucune branche distante entire/checkpoints/v1 trouvée lors du contrôle. Ne pas activer le push automatique sans accord.

La recherche distante a été réellement essayée : `entire search 'checkpoint' --json --compact --limit 1` renvoie non authentifié. Onboarding web et indexation ne sont donc pas validés. Une connexion ne garantit pas à elle seule que l'historique est publié/indexé.

## Vérifications après connexion/revue

Depuis le dépôt code : `entire auth status`, `entire doctor`, `entire status`, `entire checkpoint list`. Après publication autorisée, tester une recherche ciblée en JSON ; absence de résultats/indexation doit rester visible. Pour les nouvelles sessions Hermes, `/reload-skills` ou nouvelle session si la découverte n'est pas actualisée.

Sur un autre poste : récupérer le code contenant les skills, installer le CLI, réinstaller les hooks locaux et les approuver dans l'agent. Les hooks Git/permissions/comptes ne se transmettent pas simplement par clone. Un skill qui permet de lire l'historique ne suffit pas à activer la capture d'un agent non pris en charge.

## Sources

- https://github.com/entireio/skills
- https://entire.io/gh/entireio/skills/session/019dc0d3-6b22-7052-b168-4487a38e0f4d (lien de tutoriel fourni, contenu web non vérifié ici)
- PDF local `Entire x EHL Zurich.pdf`, texte extrait : checkpoints reliant session et commit ; le PDF ne constitue pas une preuve de configuration de notre compte.
- https://hermes-agent.nousresearch.com/docs/user-guide/features/skills/ : mécanisme documenté external_dirs.
