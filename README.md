# MP4 to EDL/SRT Converter

MP4動画ファイルから音声を抽出し、Whisperを使用して音声を文字起こしし、EDL（Edit Decision List）とSRT（字幕）ファイルを生成するPythonアプリケーションです。

## 機能

- MP4ファイルから音声を抽出
- Whisperを使用した高精度な音声文字起こし
- **音声前処理（ノイズ除去と音量正規化）**
- 無音部分を検出して音声をセグメント化
- CMX 3600形式のEDLファイル生成
- SRT形式の字幕ファイル生成
- **EDLと字幕の完全同期**（字幕がビデオクリップと正確に一致）
- **カスタム初期プロンプト**（文字起こし精度向上）
- **最適化されたWhisperパラメータ**（精度向上）
- **ビデオとオーディオの同期**（DaVinci Resolveでの完全互換性）
- 使いやすいGUIインターフェース

## 必要条件

- Python 3.7以上
- FFmpeg（システムにインストールされていること）
- 以下のPythonパッケージ:
  - openai-whisper
  - pydub
  - ffmpeg-python
  - tkinter（通常はPythonに同梱）

## インストール

1. このリポジトリをクローンまたはダウンロードします。
2. 必要なパッケージをインストールします：

```bash
pip install openai-whisper pydub ffmpeg-python
```

3. FFmpegをインストールします（まだインストールしていない場合）：
   - Windows: [FFmpeg公式サイト](https://ffmpeg.org/download.html)からダウンロードし、PATHに追加します。
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`（Ubuntuの場合）

## 使い方

### GUIモード（推奨）

1. 以下のいずれかの方法でGUIを起動します：
   - Windows: `run_gui.bat`をダブルクリック
   - macOS/Linux: ターミナルで`./run_gui.sh`を実行（必要に応じて`chmod +x run_gui.sh`で実行権限を付与）
   - または、コマンドラインで`python mp4_to_edl_srt/gui.py`を実行

2. GUIが起動したら：
   - 「入力フォルダ」に処理したいMP4ファイルが含まれるフォルダを指定します。
   - 「出力フォルダ」にEDLとSRTファイルを保存するフォルダを指定します。
   - 「初期プロンプト」に文字起こしの精度を向上させるためのプロンプトを入力します（オプション）。
   - 「変換開始」ボタンをクリックして処理を開始します。
   - 処理の進行状況はログエリアに表示されます。

### コマンドラインモード

```bash
python mp4_to_edl_srt/main.py --input <入力フォルダ> --output <出力フォルダ>
```

環境変数`WHISPER_INITIAL_PROMPT`を設定することで、初期プロンプトを指定できます：

```bash
# Windowsの場合
set WHISPER_INITIAL_PROMPT="日本語での技術的な会話"
python mp4_to_edl_srt/main.py --input <入力フォルダ> --output <出力フォルダ>

# macOS/Linuxの場合
export WHISPER_INITIAL_PROMPT="日本語での技術的な会話"
python mp4_to_edl_srt/main.py --input <入力フォルダ> --output <出力フォルダ>
```

## 出力ファイル

- `output.edl`: CMX 3600形式のEDLファイル（DaVinci Resolveなどの編集ソフトで使用可能）
- `output.srt`: SRT形式の字幕ファイル

## 音声前処理について

このアプリケーションは、文字起こしの精度を向上させるために音声ファイルに前処理を適用します：

1. **音量の正規化**：
   - 音声の音量レベルを最適化して、小さすぎる音や大きすぎる音を調整
   - pydubライブラリの`normalize`機能を使用
   - headroom=0.5の設定で、より明瞭な音声に調整

2. **ノイズ除去**：
   - 背景ノイズを軽減するためのフィルターを適用
   - FFmpegの高度なフィルターを使用
   - ハイパスフィルター（80Hz以下の低周波ノイズを除去）
   - ローパスフィルター（8000Hz以上の高周波ノイズを除去）
   - FFTノイズ除去（全体的な背景ノイズを軽減、強度-20dB）

3. **前処理の効果**：
   - 背景ノイズが多い環境での録音の文字起こし精度が向上
   - 音量が小さい部分の認識率が向上
   - 全体的な文字起こしの品質が向上
   - 人間の声の周波数帯域（80Hz〜8000Hz）を保持しながらノイズを除去

## Whisperパラメータの最適化

このアプリケーションでは、Whisperの文字起こし精度を向上させるために以下のパラメータを最適化しています：

1. **temperature (0.2)**：
   - 少し柔軟性を持たせることで、微妙な発音の揺れに対応
   - 完全に確定的な出力（0.0）よりも自然な文字起こしが可能

2. **beam_size (5)**：
   - ビームサーチの幅を広げて、より多くの候補から最適な結果を選択
   - 特に曖昧な発音や複数の解釈が可能な場合に効果的

3. **condition_on_previous_text (True)**：
   - 前後の文脈を考慮して文字起こしを行うことで、全体の一貫性が向上
   - 特に長い会話や連続した文章の文字起こしに効果的

4. **language ("ja")**：
   - 日本語に特化した処理を強制的に適用
   - 言語検出の誤りを防止

## 初期プロンプトについて

初期プロンプト機能を使用すると、Whisperモデルに文脈情報を提供し、文字起こしの精度を向上させることができます：

1. **初期プロンプトとは**：
   - Whisperモデルに与える追加の文脈情報
   - 音声の内容や話題に関するヒントを提供
   - 特定の専門用語や固有名詞の認識精度を向上

2. **効果的な初期プロンプトの例**：
   - 一般的な会話: `"日本語での自然な会話。文脈に応じて適切な表現を使用してください。"`
   - 技術的な内容: `"コンピュータ技術に関する日本語の会話。プログラミング用語が含まれます。"`
   - ビジネス会議: `"ビジネスミーティングの日本語での議事録。企業名や専門用語が含まれます。"`
   - 医療関連: `"医療に関する日本語の会話。医学用語や薬品名が含まれます。"`

3. **カスタマイズ方法**：
   - GUIの「初期プロンプト」フィールドに直接入力
   - コマンドライン使用時は環境変数`WHISPER_INITIAL_PROMPT`を設定

## EDLと字幕の同期について

このアプリケーションは、EDLと字幕（SRT）ファイルを完全に同期させる機能を持っています：

1. **同期の仕組み**：
   - 音声認識によって生成されたセグメントからEDLを生成
   - EDLのレコードタイムコード（編集タイムライン上の位置）を使用して字幕を生成
   - これにより、字幕がビデオクリップと完全に一致

2. **DaVinci Resolveでの使用方法**：
   - EDLをインポートしてビデオクリップを配置
   - SRTをインポートして字幕トラックを作成
   - 字幕がビデオクリップと完全に同期（空白やずれなし）

3. **メリット**：
   - 字幕とビデオの間に空白やずれが発生しない
   - 編集作業が効率化される
   - 字幕の位置調整が不要

## オーディオトラックについて

このアプリケーションは、ビデオとオーディオを同期させたEDLファイルを生成します：

1. **トラック指定**：
   - `AA/V`形式を使用して、ビデオとステレオオーディオを同時に指定
   - DaVinci Resolveで正しく認識されるフォーマットを採用

2. **オーディオチャンネル**：
   - ステレオオーディオ（CH1とCH2）を保持
   - 元のMP4ファイルのオーディオトラックを維持

3. **DaVinci Resolveでの表示**：
   - ビデオトラック（V1）にビデオクリップが配置
   - オーディオトラック（A1, A2）に対応するオーディオクリップが配置
   - 字幕トラック（ST1）に字幕が配置

## Whisperモデルについて

このアプリケーションはデフォルトで「base」サイズのWhisperモデルを使用しています。モデルサイズは以下の中から選択できます：

- `tiny`: 最小サイズ、最速だが精度は低い
- `base`: 小サイズ、高速で十分な精度（デフォルト）
- `small`: 中サイズ、バランスの取れた速度と精度
- `medium`: 大サイズ、高精度だが処理が遅い
- `large`: 最大サイズ、最高精度だが処理が非常に遅い

モデルサイズを変更するには、`mp4_file.py`の`transcribe`メソッド内の以下の行を編集します：

```python
model = whisper.load_model("base", device="cpu")  # "tiny", "small", "medium", "large"に変更可能
```

## CPU処理について

- このアプリケーションはCPUで処理を行います。
- 「FP16 is not supported on CPU; using FP32 instead」という警告は正常で、無視して問題ありません。
- 処理時間は入力ファイルのサイズと数、およびコンピュータの性能に依存します。

## 注意事項

- 処理時間は入力ファイルのサイズと数、およびコンピュータの性能に依存します。
- Whisperの文字起こし精度は音声の品質に依存します。
- 30fps非ドロップフレームを前提にタイムコードを計算しています。
- 大きなファイルや多数のファイルを処理する場合は、十分なメモリが必要です。
- 最適化されたパラメータ（beam_size=5など）により処理時間が長くなる場合があります。

## トラブルシューティング

- FFmpegが見つからないエラーが表示される場合は、FFmpegが正しくインストールされ、PATHに追加されていることを確認してください。
- メモリエラーが発生する場合は、より小さなMP4ファイルで試すか、コンピュータのメモリを増設してください。
- 処理が遅い場合は、より小さいWhisperモデル（tinyやbase）を使用するか、beam_sizeを小さくすることを検討してください。
- 「ファイルが見つかりません」エラーが表示される場合は、入力フォルダに.mp4ファイルが存在することを確認してください。
- 文字化けが発生する場合は、システムの言語設定を確認してください。
- 字幕とビデオのタイミングがずれる場合は、DaVinci Resolveのプロジェクト設定（フレームレート）が30fpsに設定されていることを確認してください。
- 文字起こしの精度が低い場合は、より適切な初期プロンプトを設定するか、より大きなWhisperモデルを使用してください。
- 音声前処理中にエラーが発生する場合は、前処理をスキップして元の音声ファイルが使用されます。
- その他の問題が発生した場合は、ログを確認して問題を特定してください。
