import { ImageResponse } from "next/og";

const SUPPORTED_SIZES = new Set([180, 192, 512]);

export const runtime = "edge";

export async function GET(
  _request: Request,
  context: { params: Promise<{ size: string }> },
): Promise<Response> {
  const { size: rawSize } = await context.params;
  const size = Number(rawSize);
  if (!SUPPORTED_SIZES.has(size)) return new Response("Unsupported icon size", { status: 404 });
  const padding = Math.round(size * 0.19);
  const inner = size - padding * 2;
  return new ImageResponse(
    <div
      style={{
        alignItems: "center",
        background: "#18211e",
        display: "flex",
        height: "100%",
        justifyContent: "center",
        width: "100%",
      }}
    >
      <svg width={inner} height={inner} viewBox="0 0 512 512">
        <path d="M256 65 472 447H40L256 65Z" fill="#f3f5f2" />
        <path d="M256 169 355 350H157L256 169Z" fill="#176c63" />
        <circle cx="256" cy="294" r="39" fill="#d65d3b" />
      </svg>
    </div>,
    { width: size, height: size },
  );
}
