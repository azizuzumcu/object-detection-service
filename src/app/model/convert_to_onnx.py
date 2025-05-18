import torch
import onnx


def convert_yolo_pt_to_onnx(
    pt_path: str,
    onnx_path: str,
    input_size: tuple = (640, 640),
    dynamic_axes: bool = True,
):
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=pt_path, force_reload=True
    )
    model.eval()

    dummy_input = torch.zeros((1, 3, input_size[1], input_size[0]), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=12,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=(
            {
                "images": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size"},
            }
            if dynamic_axes
            else None
        ),
        do_constant_folding=True,
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model başarıyla oluşturuldu: {onnx_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert YOLO .pt to ONNX")
    parser.add_argument("--pt", required=True, help="Path to YOLO .pt file")
    parser.add_argument("--onnx", required=True, help="Output path for .onnx file")
    parser.add_argument("--width", type=int, default=640, help="Input width")
    parser.add_argument("--height", type=int, default=640, help="Input height")
    parser.add_argument(
        "--no-dynamic", action="store_true", help="Disable dynamic axes"
    )
    args = parser.parse_args()

    convert_yolo_pt_to_onnx(
        pt_path=args.pt,
        onnx_path=args.onnx,
        input_size=(args.width, args.height),
        dynamic_axes=not args.no_dynamic,
    )
