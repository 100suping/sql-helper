import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from transformers import BitsAndBytesConfig


def test_qwen_model():
    try:
        print("=== Qwen 모델 테스트 시작 ===")

        # 모델 설정
        model_name = "100suping/Qwen2.5-Coder-34B-Instruct-kosql-adapter"
        huggingface_token = "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"

        print(f"모델 로딩 시작: {model_name}")

        # 8비트 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16
        )

        # 모델과 토크나이저 로드
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            token=huggingface_token,
            quantization_config=bnb_config,
        )

        # 추론을 위한 모델 준비 - 이 부분이 추가됨
        model = FastLanguageModel.for_inference(model)

        print("모델과 토크나이저 로드 완료")

        # 테스트용 입력
        test_input = "전체 주문 중 환불된 주문의 비율은 얼마인가요?"
        print(f"\n테스트 입력: {test_input}")

        # 입력 토크나이징
        inputs = tokenizer(test_input, return_tensors="pt")
        print("토크나이징 완료")

        # 모델 추론
        print("\n추론 시작...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 결과 디코딩
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n=== 결과 ===")
        print(result)

        return True

    except Exception as e:
        print("\n=== 에러 발생 ===")
        print(f"에러 타입: {type(e)}")
        print(f"에러 메시지: {str(e)}")
        return False


if __name__ == "__main__":
    print("CUDA 사용 가능:", torch.cuda.is_available())
    print("사용 가능한 GPU:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("현재 GPU:", torch.cuda.get_device_name(0))

    success = test_qwen_model()
    print("\n테스트 결과:", "성공" if success else "실패")
