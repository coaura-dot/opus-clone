# Política de Privacidade — Auto Clipper

*Última atualização: 6 de outubro de 2026*

O **Auto Clipper** é uma ferramenta de uso pessoal que roda no computador de
quem a usa. Ela edita cortes verticais curtos de vídeos e, se o usuário
autorizar, envia esses cortes para o **próprio canal do YouTube do usuário**
pela YouTube Data API v3 (YouTube API Services).

## Termos do YouTube e do Google

Ao autorizar o Auto Clipper a acessar a sua conta do YouTube, você concorda
com os [Termos de Serviço do YouTube](https://www.youtube.com/t/terms). O uso
dos dados pelo Google é regido pela
[Política de Privacidade do Google](https://policies.google.com/privacy).

## Quais dados são acessados

Com a sua autorização (login OAuth do Google), o Auto Clipper usa:

- **youtube.upload**: para enviar ao seu canal os vídeos que a ferramenta gerou;
- **youtube.readonly**: só para ler o nome do seu próprio canal e confirmar
  em qual canal os vídeos serão postados;
- **youtube**: só para mudar a visibilidade (privado → público) dos vídeos que
  a própria ferramenta enviou ao seu canal.

A ferramenta **não** lê, coleta ou guarda dados de outros usuários, de outros
canais, comentários, inscritos ou estatísticas.

## Onde os dados ficam

- O token de acesso fica salvo **apenas no seu computador**
  (`credentials/youtube_token.json`). Ele não é enviado a nenhum servidor do
  Auto Clipper; não existe servidor do Auto Clipper.
- A ferramenta guarda localmente só os IDs dos vídeos que ela mesma enviou
  (`autopilot_data/state.json`), para não postar o mesmo corte duas vezes.
- Nenhum dado é vendido, compartilhado com terceiros nem usado para
  publicidade. Os únicos serviços contatados são os do Google/YouTube.

## Como remover o acesso e os dados

- Revogue o acesso a qualquer momento em
  [Permissões da Conta do Google](https://myaccount.google.com/permissions)
  ou na página de
  [segurança da conta](https://security.google.com/settings/security/permissions).
- Para apagar os dados locais, exclua as pastas `credentials/` e
  `autopilot_data/` da pasta do programa.

## Contato

Dúvidas sobre esta política: abra uma issue em
<https://github.com/coaura-dot/opus-clone/issues>.

---

# Privacy Policy — Auto Clipper (English)

Auto Clipper is a personal tool that runs on the user's own computer. It edits
short vertical clips and, when the user authorizes it, uploads them to the
**user's own YouTube channel** through the YouTube Data API v3 (YouTube API
Services). By authorizing it you agree to the
[YouTube Terms of Service](https://www.youtube.com/t/terms); see also the
[Google Privacy Policy](https://policies.google.com/privacy).

- Scopes: `youtube.upload` (upload the generated videos to your channel),
  `youtube.readonly` (read your own channel name only) and `youtube` (only to
  change the visibility, private to public, of videos the tool itself uploaded).
- No data about other users or channels is accessed. The OAuth token is stored
  only on your computer; there is no Auto Clipper server. Only the IDs of
  videos the tool uploaded are kept locally, to avoid duplicates. No data is
  sold, shared or used for advertising.
- Revoke access at <https://myaccount.google.com/permissions>; delete the
  local `credentials/` and `autopilot_data/` folders to remove local data.
- Contact: <https://github.com/coaura-dot/opus-clone/issues>.
