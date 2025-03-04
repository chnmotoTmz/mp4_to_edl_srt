import os
import subprocess
import whisper
import pydub
import re
import json
import torch  # torchを明示的にインポート
from typing import List, Dict, Tuple, Optional, Any
import warnings
from pydub import AudioSegment
from pydub.silence import split_on_silence
from datetime import datetime, timedelta

# faster-whisperのサポートを追加（インストールされていない場合はスキップ）
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    print("faster-whisperライブラリが利用可能です。高速モードが使用できます。")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("faster-whisperライブラリが見つかりません。標準モードで実行します。")
    print("高速モードを使用するには次のコマンドを実行してください: pip install faster-whisper")

from segment import Segment
from edl_data import EDLData
from srt_data import SRTData

# Whisperの警告を非表示にする
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

class MP4File:
    def __init__(self, filepath: str, file_index: int):
        self.filepath: str = os.path.normpath(filepath)
        self.file_index: int = file_index  # For reel name (e.g., TAPE01)
        self.audio_filepath: str = ""
        self.transcription_result: Dict = {}
        self.segments: List[Segment] = []
        self.edl_data: EDLData = EDLData(title="My Video Project", fcm="NON-DROP FRAME")
        self.srt_data: SRTData = SRTData()
        self.creation_time: Optional[str] = None
        self.timecode_offset: Optional[str] = None
        
        # ファイルのメタデータを抽出
        self.extract_metadata()

    def extract_metadata(self) -> None:
        """MP4ファイルからメタデータ（作成時間やタイムコード）を抽出します。"""
        try:
            print(f"ファイルのメタデータを抽出中: {self.filepath}")
            
            # FFprobeを使用してメタデータを抽出
            command = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                self.filepath
            ]
            
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            metadata = json.loads(result.stdout)
            
            # creation_timeを探す
            if "format" in metadata and "tags" in metadata["format"]:
                tags = metadata["format"]["tags"]
                if "creation_time" in tags:
                    self.creation_time = tags["creation_time"]
                    print(f"ファイル作成時間: {self.creation_time}")
                    
                    # creation_timeからタイムコードオフセットを計算
                    try:
                        # ISO 8601形式の日時文字列をパース (例: 2023-01-01T12:00:00.000Z)
                        dt = datetime.fromisoformat(self.creation_time.replace('Z', '+00:00'))
                        # 時間部分だけを取得してタイムコードに変換
                        hours = dt.hour
                        minutes = dt.minute
                        seconds = dt.second
                        frames = int(dt.microsecond / 1000000 * 30)  # 30fpsとして計算
                        
                        self.timecode_offset = f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
                        print(f"計算されたタイムコードオフセット: {self.timecode_offset}")
                    except Exception as e:
                        print(f"タイムコードオフセットの計算中にエラーが発生しました: {e}")
            
            # タイムコードトラックを探す
            for stream in metadata.get("streams", []):
                if stream.get("codec_type") == "video" and "tags" in stream:
                    tags = stream["tags"]
                    if "timecode" in tags:
                        # タイムコードトラックが存在する場合はそれを優先
                        self.timecode_offset = tags["timecode"]
                        print(f"ビデオストリームからタイムコードを検出: {self.timecode_offset}")
                        break
            
            if not self.timecode_offset:
                print(f"警告: タイムコードが検出されませんでした。デフォルトの00:00:00:00を使用します。")
                self.timecode_offset = "00:00:00:00"
                
        except subprocess.CalledProcessError as e:
            print(f"FFprobeエラー: {e.stderr}")
            print(f"警告: メタデータの抽出に失敗しました。デフォルトのタイムコードを使用します。")
            self.timecode_offset = "00:00:00:00"
        except Exception as e:
            print(f"メタデータ抽出中にエラーが発生しました: {str(e)}")
            print(f"警告: デフォルトのタイムコードを使用します。")
            self.timecode_offset = "00:00:00:00"

    def apply_timecode_offset(self, timecode: str) -> str:
        """
        タイムコードにオフセットを適用します。
        
        Args:
            timecode: 元のタイムコード (HH:MM:SS:FF形式)
            
        Returns:
            オフセットが適用されたタイムコード
        """
        if not self.timecode_offset or self.timecode_offset == "00:00:00:00":
            return timecode
            
        # タイムコードを秒に変換
        tc_seconds = self._timecode_to_seconds(timecode)
        offset_seconds = self._timecode_to_seconds(self.timecode_offset)
        
        # オフセットを適用
        result_seconds = tc_seconds + offset_seconds
        
        # 秒をタイムコードに戻す
        return self._seconds_to_timecode(result_seconds)

    def extract_audio(self) -> None:
        """Extracts audio from the MP4 file using FFmpeg."""
        base_name = os.path.splitext(os.path.basename(self.filepath))[0]
        output_dir = os.path.dirname(self.filepath)
        self.audio_filepath = os.path.join(output_dir, f"{base_name}.wav")
        
        command = [
            "ffmpeg",
            "-i", self.filepath,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-y",  # Overwrite output file if it exists
            self.audio_filepath,
        ]
        try:
            print(f"FFmpegコマンド: {' '.join(command)}")
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"音声ファイルを抽出しました: {self.audio_filepath}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpegエラー: {e.stderr}")
            raise Exception(f"FFmpeg error: {e.stderr}")

    def transcribe(self, initial_prompt: str = None) -> None:
        """Transcribes the audio file using Whisper with word-level timestamps."""
        try:
            if not os.path.exists(self.audio_filepath):
                raise FileNotFoundError(f"音声ファイルが見つかりません: {self.audio_filepath}")
            
            # PyTorchの警告を抑制
            warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
            
            # GPUメモリの最適化設定
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
            
            # 音声前処理の有効/無効を環境変数から取得
            enable_preprocessing = os.environ.get("ENABLE_AUDIO_PREPROCESSING", "True").lower() == "true"
            
            # 音声ファイルの前処理（オプション）
            audio_file_to_use = self.audio_filepath
            if enable_preprocessing:
                print("音声前処理を実行します...")
                audio_file_to_use = self._preprocess_audio(self.audio_filepath)
            else:
                print("音声前処理をスキップします - 元の音声ファイルを使用")
            
            # 初期プロンプトが指定されていない場合はデフォルト値を使用
            if initial_prompt is None:
                # より控えめな初期プロンプトを使用
                initial_prompt = ""  # 空のプロンプトに変更
                
            print(f"文字起こし中: {audio_file_to_use}")
            print(f"使用する初期プロンプト: {initial_prompt}")

            # faster-whisperが利用可能で、高速モードが選択されている場合
            use_faster_whisper = FASTER_WHISPER_AVAILABLE
            
            if use_faster_whisper:
                print("faster-whisperを使用して文字起こしを実行します（高速モード）")
                # CTranslate2最適化されたモデルを使用
                try:
                    # 高速なwhisperモデルを使用 - 常にCPUで実行
                    model = WhisperModel(
                        model_size_or_path="deepdml/faster-whisper-large-v3-turbo-ct2", 
                        device="cpu",  # 常にCPUを使用 
                        compute_type="int8"  # int8量子化で効率的に実行
                    )
                    
                    print("モデル情報: large-v3-turbo (CPU, int8量子化)")
                    
                    # faster-whisperのTranscribeオプション
                    segments, info = model.transcribe(
                        audio_file_to_use,
                        language="ja",
                        task="transcribe",
                        initial_prompt=initial_prompt,
                        condition_on_previous_text=True,
                        temperature=0.2,
                        beam_size=5,
                        word_timestamps=True,
                        vad_filter=True,  # 音声区間検出フィルタを有効化
                        vad_parameters={"min_silence_duration_ms": 500}  # 無音区間のパラメータ
                    )
                    
                    print(f"検出された言語: {info.language} (確度: {info.language_probability:.2f})")
                    
                    # セグメントを保存
                    self.segments = []
                    for segment in segments:
                        words = []
                        if hasattr(segment, 'words') and segment.words:
                            for word in segment.words:
                                words.append({
                                    "word": word.word,
                                    "start": word.start,
                                    "end": word.end,
                                    "probability": word.probability
                                })
                        
                        # faster-whisperのセグメントをアプリケーション形式に変換
                        # start -> start_timecode, end -> end_timecode, text -> transcription
                        start_timecode = self._seconds_to_timecode(segment.start)
                        end_timecode = self._seconds_to_timecode(segment.end)
                        self.segments.append(Segment(
                            start_timecode=start_timecode,
                            end_timecode=end_timecode,
                            transcription=segment.text
                        ))
                    
                    # 結果を保存
                    self.transcription_result = {
                        "text": " ".join([segment.text for segment in segments]),
                        "segments": [{
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text
                        } for segment in segments],
                        "language": info.language
                    }
                    
                    # EDLとSRTデータを生成
                    self.generate_edl_data("00:00:00:00")
                    self.generate_srt_data()
                    return
                    
                except Exception as e:
                    print(f"faster-whisperでのエラー: {str(e)}")
                    print("標準のwhisperにフォールバックします...")
            
            # 標準のwhisperを使用する場合は、ここから下の既存のコードを実行
            # 勾配計算を無効化
            torch.set_grad_enabled(False)
            
            # CUDA初期化前の設定
            if torch.cuda.is_available():
                # GPUキャッシュをクリア
                torch.cuda.empty_cache()
                # メモリの断片化を防ぐ
                torch.cuda.memory.set_per_process_memory_fraction(0.8)  # GPUメモリの80%まで使用に制限
                # CUDAストリームを同期
                torch.cuda.synchronize()
                
                # CUBLASワークスペースを制限
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                
                # 利用可能なGPUメモリをチェック
                total_memory = torch.cuda.get_device_properties(0).total_memory
                available_memory = total_memory - torch.cuda.memory_allocated(0)
                print(f"利用可能なGPUメモリ: {available_memory / 1024**3:.2f} GB")
            
            # モデルロードの関数を定義
            def load_model_safely(model_name, device):
                try:
                    # 古いバージョンのWhisperでは一部のパラメータがサポートされていないため削除
                    model = whisper.load_model(
                        model_name,
                        device=device,
                        download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisper")
                    )
                    return model
                except Exception as e:
                    print(f"モデルロード中のエラー: {str(e)}")
                    return None
            
            # モデルサイズの要件（より現実的な見積もり）
            model_memory_requirements = {
                "medium": 5 * 1024**3,     # 5GB
                "small": 2 * 1024**3       # 2GB
            }
            
            # モデル選択
            model = None
            
            if torch.cuda.is_available():
                # 利用可能なメモリに基づいてモデルを選択
                for model_name, required_memory in model_memory_requirements.items():
                    if available_memory >= required_memory * 1.2:  # 20%のバッファを追加
                        print(f"選択したモデル: {model_name} (必要メモリ: {required_memory / 1024**3:.1f}GB)")
                        model = load_model_safely(model_name, "cuda")
                        if model is not None:
                            print(f"{model_name}モデルのロードに成功しました")
                            break
                        else:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
            
            # どのモデルもロードできなかった場合、CPUでsmallモデルを使用
            if model is None:
                print("警告: GPUモデルのロードに失敗したため、CPUでsmallモデルを使用します")
                model = load_model_safely("small", "cpu")
                if model is None:
                    raise Exception("すべてのモデルのロードに失敗しました")
                print("smallモデルをCPUにロードしました")
            
            # 文字起こしを実行
            transcribe_options = {
                "language": "ja",
                "task": "transcribe",
                "temperature": 0.2,
                "beam_size": 5,
                "initial_prompt": initial_prompt,
                "condition_on_previous_text": True,
                "verbose": True,
                "fp16": False,  # 精度を優先
                "suppress_tokens": [-1]  # 特殊トークンを抑制
            }
            
            # 文字起こしを実行
            result = model.transcribe(audio_file_to_use, **transcribe_options)
            
            self.transcription_result = result
            
            print(f"文字起こし完了: {len(self.transcription_result.get('segments', []))}セグメント")
        except Exception as e:
            print(f"文字起こし中にエラーが発生しました: {str(e)}")
            raise Exception(f"Whisper transcription error: {e}")

    def _preprocess_audio(self, audio_path: str) -> str:
        """音声ファイルの前処理（ノイズ除去と音量調整）を行います。"""
        try:
            print(f"音声ファイルを前処理中: {audio_path}")
            
            # 出力ファイルパスを生成
            base_dir = os.path.dirname(audio_path)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            processed_path = os.path.join(base_dir, f"{base_name}_processed.wav")
            
            # pydubを使用して音声を読み込み
            audio = AudioSegment.from_file(audio_path)
            
            # 音量の正規化（headroomを0.5に調整してより大きな音量に）
            print(f"音量を正規化中...")
            normalized_audio = audio.normalize(headroom=0.5)
            
            # ノイズ除去（FFmpegを使用）
            print(f"ノイズ除去フィルターを適用中...")
            normalized_audio.export("temp_normalized.wav", format="wav")
            
            # FFmpegのノイズ除去フィルターを適用（周波数帯域を拡大し、ノイズ除去強度を調整）
            command = [
                "ffmpeg",
                "-i", "temp_normalized.wav",
                "-af", "highpass=f=80,lowpass=f=8000,afftdn=nf=-20",  # 元のフィルター設定
                "-y",  # 既存ファイルを上書き
                processed_path
            ]
            
            subprocess.run(command, check=True, capture_output=True, text=True)
            
            # 一時ファイルを削除
            if os.path.exists("temp_normalized.wav"):
                os.remove("temp_normalized.wav")
                
            print(f"音声前処理完了: {processed_path}")
            return processed_path
            
        except Exception as e:
            print(f"音声前処理中にエラーが発生しました: {str(e)}")
            print(f"前処理をスキップして元の音声ファイルを使用します。")
            return audio_path

    def segment_audio(self, threshold: float = 0.5) -> None:
        """Segments the audio based on silence threshold using Whisper timestamps."""
        segments = self.transcription_result.get("segments", [])
        if not segments:
            print("警告: 文字起こし結果にセグメントが含まれていません")
            return

        # Whisperのセグメントをそのまま使用する
        print(f"音声をセグメント化中...")
        
        # タイムコードでソート
        sorted_segments = sorted(segments, key=lambda x: x.get("start", 0))
        
        for i, segment in enumerate(sorted_segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            # 空のセグメントをスキップ
            if not text:
                continue
                
            # 極端に短いセグメントをスキップ（0.2秒未満）
            if end_time - start_time < 0.2:
                continue
                
            # タイムコードの逆転がないか確認
            if i > 0 and start_time < sorted_segments[i-1].get("end", 0):
                # 前のセグメントの終了時間より新しい開始時間を設定
                start_time = sorted_segments[i-1].get("end", 0) + 0.1
                
            start_timecode = self._seconds_to_timecode(start_time)
            end_timecode = self._seconds_to_timecode(end_time)
            
            # セグメントを追加
            self.segments.append(Segment(start_timecode, end_timecode, text))
            print(f"セグメント追加: {start_timecode} - {end_timecode} | {text[:30]}...")

    def _add_segment(self, words: List[Dict]) -> None:
        """Creates a Segment from a list of words."""
        start_time = words[0]["start"]
        end_time = words[-1]["end"]
        # 単語間のスペースを調整（日本語の場合は不要）
        language = self.transcription_result.get("language", "")
        if language in ["ja", "zh", "ko"]:
            # 日本語、中国語、韓国語の場合はスペースを削除
            transcription = "".join(word["word"].strip() for word in words)
        else:
            # その他の言語の場合は単語間にスペースを入れる
            transcription = " ".join(word["word"].strip() for word in words)
            
        start_timecode = self._seconds_to_timecode(start_time)
        end_timecode = self._seconds_to_timecode(end_time)
        self.segments.append(Segment(start_timecode, end_timecode, transcription))
        print(f"セグメント追加: {start_timecode} - {end_timecode}")

    def generate_edl_data(self, record_start: str, use_timecode_offset: bool = True) -> Tuple[EDLData, str]:
        """Generates EDL data with sequential record timecodes."""
        reel_name = f"TAPE{self.file_index:02d}"
        current_record = self._timecode_to_seconds(record_start)

        print(f"EDLデータを生成中 (リール名: {reel_name}, 開始レコード: {record_start})...")
        print(f"タイムコードオフセットの使用: {'有効' if use_timecode_offset else '無効'}")
        
        # EDLイベントのリストをクリア
        self.edl_data = EDLData(title="My Video Project", fcm="NON-DROP FRAME")
        
        # EDLイベントとそのレコードタイムコードを保存するリスト
        self.edl_events_with_timecode = []
        
        # 内部タイムコードの終了時間を計算（開始時間 + 1時間と仮定）
        if self.timecode_offset and self.timecode_offset != "00:00:00:00":
            start_time_seconds = self._timecode_to_seconds(self.timecode_offset)
            end_time_seconds = start_time_seconds + 3600  # 1時間後
            print(f"内部タイムコード範囲: {self.timecode_offset} - {self._seconds_to_timecode(end_time_seconds)}")
        
        for segment in self.segments:
            # タイムコードオフセットを適用（オプション）
            if use_timecode_offset and self.timecode_offset:
                source_in = self.apply_timecode_offset(segment.start_timecode)
                source_out = self.apply_timecode_offset(segment.end_timecode)
                
                # 内部タイムコードの範囲をチェック
                source_out_seconds = self._timecode_to_seconds(source_out)
                if source_out_seconds > end_time_seconds:
                    print(f"警告: セグメントの終了時間 ({source_out}) が内部タイムコードの範囲を超えています。終了時間を調整します。")
                    source_out = self._seconds_to_timecode(end_time_seconds)
                    
                print(f"タイムコードオフセット適用: {segment.start_timecode} → {source_in}")
            else:
                source_in = segment.start_timecode
                source_out = segment.end_timecode
            
            duration = self._timecode_to_seconds(source_out) - self._timecode_to_seconds(source_in)
            record_in = self._seconds_to_timecode(current_record)
            record_out = self._seconds_to_timecode(current_record + duration)
            
            # ビデオとオーディオを含むイベント（DaVinci Resolveで正しく認識される形式）
            main_event = {
                "reel_name": reel_name,
                "track_type": "AA/V",  # オーディオとビデオの両方
                "transition": "C",
                "source_in": source_in,
                "source_out": source_out,
                "record_in": record_in,
                "record_out": record_out,
                "clip_name": os.path.basename(self.filepath),
            }
            self.edl_data.add_event(main_event)
            
            # EDLイベントとそのレコードタイムコードを保存（SRT生成用）
            self.edl_events_with_timecode.append({
                "segment": segment,
                "record_in": record_in,
                "record_out": record_out
            })
            
            current_record += duration

        new_record_start = self._seconds_to_timecode(current_record)
        print(f"EDLデータ生成完了: {len(self.edl_data.events)}イベント, 次の開始レコード: {new_record_start}")
        return self.edl_data, new_record_start

    def generate_srt_data(self) -> SRTData:
        """Generates SRT data based on EDL record timecodes."""
        print(f"SRTデータを生成中...")
        
        # SRTデータを初期化
        self.srt_data = SRTData()
        
        # EDLのレコードタイムコードに基づいてSRTセグメントを生成
        if hasattr(self, 'edl_events_with_timecode') and self.edl_events_with_timecode:
            print(f"EDLのレコードタイムコードに基づいてSRTデータを生成します")
            
            for event in self.edl_events_with_timecode:
                segment = event["segment"]
                record_in = event["record_in"]
                record_out = event["record_out"]
                
                # EDLのレコードタイムコードを使用して新しいセグメントを作成
                srt_segment = Segment(
                    record_in,
                    record_out,
                    segment.transcription
                )
                
                self.srt_data.add_segment(srt_segment)
        else:
            # EDLデータがない場合は元のセグメントを使用
            print(f"警告: EDLデータが見つかりません。元のセグメントを使用します。")
            for segment in self.segments:
                self.srt_data.add_segment(segment)
        
        print(f"SRTデータ生成完了: {len(self.srt_data.segments)}セグメント")
        return self.srt_data

    def _seconds_to_timecode(self, seconds: float) -> str:
        """Converts seconds to HH:MM:SS:FF format (30 fps)."""
        total_frames = int(seconds * 30)
        hours = total_frames // (3600 * 30)
        minutes = (total_frames % (3600 * 30)) // (60 * 30)
        secs = (total_frames % (60 * 30)) // 30
        frames = total_frames % 30
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

    def _timecode_to_seconds(self, timecode: str) -> float:
        """Converts HH:MM:SS:FF to seconds (30 fps)."""
        hh, mm, ss, ff = map(int, timecode.split(":"))
        return hh * 3600 + mm * 60 + ss + ff / 30.0

