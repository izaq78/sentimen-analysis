# Resource Sources

## Emoji Sentiment Ranking 1.0

- Nama dataset: Emoji Sentiment Ranking 1.0
- Pembuat: Petra Kralj Novak, Jasmina Smailovic, Borut Sluban, Igor Mozetic
- Tahun: 2015
- URL sumber: https://www.clarin.si/repository/xmlui/handle/11356/1048
- Lisensi: Creative Commons Attribution-ShareAlike 4.0 International
- Jumlah entri: 751 emoji
- Tanggal unduh: dicatat otomatis di `cache/emoji_sentiment_metadata.json`
- Fungsi resource: memberi skor numerik emoji yang dikonversi ke label `positif`, `negatif`, atau `netral` memakai `resources/sentiment_config.json`.

## Translation Cache

- File: `cache/translation_cache.json`
- Sumber input: metadata nama emoji dari library `emoji` dan makna emoticon dari library `emot`
- Mesin penerjemah setup: `deep-translator`
- Tahun: 2026
- URL library emoji: https://github.com/carpedm20/emoji
- URL library emot: https://github.com/NeelShah18/emot
- Fungsi resource: menyimpan hasil terjemahan Inggris ke Bahasa Indonesia agar preprocessing tidak melakukan request internet.

## Kamus Bahasa Indonesia

- File: `kamus_indonesia.txt`
- Fungsi resource: validasi kandidat leetspeak, huruf berulang, dan koreksi typo RapidFuzz.

## Leetspeak Rules

- File: `resources/leet_rules.json`
- Fungsi resource: konfigurasi substitusi kandidat leetspeak. File ini bukan lexicon sentimen.
