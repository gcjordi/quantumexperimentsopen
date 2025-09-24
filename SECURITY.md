# Seguretat · Quantum Experiments Open

Gràcies per ajudar-nos a mantenir aquest projecte segur. Acceptem informes de vulnerabilitats seguint una política de **divulgació coordinada**.

## 📫 Com reportar
- Envia un correu a: info@jordigarcia.eu
- No obris *issues* públiques per problemes de seguretat

Inclou, si és possible:
- Descripció clara del problema i impacte
- Passos per reproduir (PoC)
- Entorn afectat (versió, OS, dependències)
- Recomanació inicial de mitigació

## ⏱️ Timeline i resposta (no garantit)
- Confirmació de recepció en **72 hores hàbils**
- Actualització d’estat en **7 dies hàbils**
- Objectiu de correcció en **30 dies hàbils** (pot variar segons la gravetat)

## 🔭 Abast
Aquest projecte és codi de recerca (VQC per classificació). Ens interessa especialment:
- Execució arbitrària de codi o escalat de privilegis
- Vulnerabilitats a dependències (p. ex. càrrega insegura, RCE, deserialització)
- Filtració de secrets (tokens, claus, credencials)
- Problemes a scripts o notebooks que puguin exposar dades locals o credencials

No considerem dins l’abast:
- Errors de rendiment no relacionats amb seguretat
- Configuracions d’usuari finals (entorns locals) fora del nostre control
- Resultats “incorrectes” de l’algorisme (no és una vulnerabilitat de seguretat)

## 🔐 Bones pràctiques recomanades
- No pujis **secrets** al repositori (usa variables d’entorn)
- Actualitza dependències regularment
- Executa el codi en **entorns aïllats** (virtualenv/conda, comptes sandbox, IAM mínims)

## 🏷️ Crèdit i reconeixement
Amb el teu permís, podem agrair públicament les troballes un cop solucionades. No oferim *bounties* en aquest moment.

