

enum ComparisonTextType {
  ALUM = 1,
  NUMS = 2,
}

const splitImage = async (f: string): Promise<CanvasRenderingContext2D[]> => {
  // im = Image.open("scores/" + f);
  const img = new Image();
  img.src = f;
  const ctxs = await new Promise<CanvasRenderingContext2D[]>(
    (resolve, reject) => {
      img.onload = () => {
        const canvas1 = document.createElement("canvas");
        const canvas2 = document.createElement("canvas");
        const l1 = 45,
          l2 = 364,
          r1 = 120,
          r2 = 420,
          t = 85,
          b = 500;
        const ctx1 = canvas1.getContext("2d");
        const ctx2 = canvas2.getContext("2d");
        if (!!ctx1 && !!ctx2) {
          ctx1.drawImage(img, l1, t, r1, b);
          ctx2.drawImage(img, l2, t, r2, b);
          resolve([ctx1, ctx2]);
        } else {
          reject();
        }
      };
    }
  );
  return ctxs;
};

const alphanumWhitelist =
  "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàáâãäåéêëìíîïóôõöòøùúûüýÿ" +
  "àáâãäåéêëìíîïóôõöòøùúûüýÿ".toUpperCase();
const numWhitelist = "0123456789";

const readImg = async (
  ctx: CanvasRenderingContext2D,
  textType: ComparisonTextType
): Promise<Map<number, { [key: string]: number }> | string[]> => {
  const worker = await createWorker("eng");

  let startRange = 3;
  let endRange = 7;
  let ret: Map<number, { [key: string]: number }> | string[] = [];
  switch (textType) {
    case ComparisonTextType.ALUM:
      await worker.setParameters({
        tessedit_char_whitelist: alphanumWhitelist,
        tessedit_pageseg_mode: "6",
      });
      break;
    case ComparisonTextType.NUMS:
      startRange = 4;
      endRange = 5;
      await worker.setParameters({
        tessedit_char_whitelist: numWhitelist,
        tessedit_pageseg_mode: "6",
      });

      break;
  }
  ret = new Map<number, { [key: string]: number }>();
  for (let iteration = startRange; iteration < endRange; iteration++) {
    ctx.scale(iteration, iteration);
    const d = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height); // Get image Data from Canvas context
    const threshold = 255;
    for (let j = 0; j < d.data.length; j += 4) {
      // 4 is for RGBA channels
      // R=G=B=R>T?255:0
      d.data[j] =
        d.data[j + 1] =
        d.data[j + 2] =
          d.data[j + 1] > threshold ? 255 : 0;
    }
    // ctx.putImageData(d, 0, 0);
    const ocrOut = await worker.recognize(d);
    const names = ocrOut.data.text.split("\n");
    for (let i = 0; i < names.length; i++) {
      const trimmed = names[i].trim();
      if (!ret.has(i)) {
        ret.set(i, {});
      }
      if (!ret.get(i)?.[trimmed]) {
        const r = ret.get(i);
        if (r) r[trimmed] = 0;
      }
      const r = ret.get(i);
      if (r) r[trimmed] = r[trimmed] + 1;
    }
  }
  await worker.terminate();
  if (textType === ComparisonTextType.NUMS) {
    for (let iteration = startRange; iteration < endRange; iteration++) {
      const r: string[] = [];
      ret.forEach((e) => {
        r.push(Object.keys(e)[0]);
      });
      return r;
    }
  }
  return ret;
};

export const ocr = async (
  base64Images: string[],
  members: string[]
): Promise<{ [key: string]: number }> => {
  const ret: { [key: string]: number } = {};
  for (const imgData of base64Images) {
    const [namesCtx, scoresCtx] = await splitImage(imgData);
    const namesOCR = await readImg(namesCtx, ComparisonTextType.ALUM);
    if (!(namesOCR instanceof Map)) {
      return {};
    }
    namesOCR.forEach((e) => {
      console.log(e);
    });
    const scoresOCR = await readImg(scoresCtx, ComparisonTextType.NUMS);
    if (!(namesOCR instanceof Array)) {
      return {};
    }
    scoresOCR.forEach((e) => {
      console.log(e);
    });
  }

  return ret;
};
