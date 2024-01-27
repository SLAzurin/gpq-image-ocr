<script setup lang="ts">
import { ref } from "vue";

defineProps();

const results = ref<any>({});

const members = ref<string[]>([]);
try {
  const storageMembers = localStorage.getItem("members");
  if (storageMembers) members.value = JSON.parse(storageMembers);
} catch (e: unknown) {
  // dont do anything, just leave it []
}
const statusStr = ref("");
const statusType = ref<"alert-success" | "alert-error" | "">("");
const processing = ref(false);
const imagesF = ref(new Array<string>());
const processImages = () => {
  console.log("processing");
};

const pasteContent = (e: ClipboardEvent) => {
  statusType.value = "";
  if (e.clipboardData?.items[0].type.startsWith("image/")) {
    const f = e.clipboardData?.items[0].getAsFile();
    if (f) {
      const r = new FileReader();
      r.readAsDataURL(f);
      r.onloadend = () => {
        const base64data = r.result;
        if (base64data) {
          imagesF.value.push(base64data?.toString());
          processImages();
        }
      };
    }
  } else if (e.clipboardData?.items[0].kind === "string") {
    e.clipboardData?.items[0].getAsString((v) => {
      try {
        const m: string[] = JSON.parse(v);
        members.value = m;
        localStorage.setItem("members", JSON.stringify(m));
        statusStr.value = `Successfully imported ${m.length} member(s)`;
        statusType.value = "alert-success";
      } catch (e: unknown) {
        statusStr.value = "Failed to update members list";
        statusType.value = "alert-error";
      }
    });
  }
};

const resetImages = () => {
  results.value = {};
  statusType.value = "";
  imagesF.value = [];
};

const copyResultsToClipboard = () => {
  try {
    navigator.clipboard.writeText(JSON.stringify(results.value));
    statusStr.value = "Copied result to clipboard!";
    statusType.value = "alert-success";
  } catch (err) {
    alert("Failed to copy results to clipboard");
  }
};
</script>

<template>
  <div class="main-run">
    <h1 style="margin-bottom: 1em">Welcome to GPQ Image OCR</h1>
    <div class="horizontal">
      <div class="vertical">
        <h4>
          Credits:<br />UI & supporting logic dev:
          <span class="underline">AzurinDayo</span> (iMonoxian)<br />Main logic
          dev: <span class="underline">qbkl</span> (inuwater)
        </h4>
        <h4>
          OtherContributors:<br />Supporting logic dev:
          <span class="underline">YellowCello</span> (BlueFlute)
        </h4>
        <p></p>
        <textarea
          name="paste-images"
          id="paste-images"
          cols="40"
          rows="10"
          :placeholder="`Paste members and images here

${members.length} member(s)
${imagesF.length} image(s)`"
          @paste="pasteContent"
          readonly
        ></textarea>
        <div v-if="processing">Working...</div>
        <div v-else></div>
        <button @click="resetImages">Reset images</button>
      </div>
      <div class="center">
        <button @click="copyResultsToClipboard">
          Copy results to clipboard
        </button>
        <pre>{{ JSON.stringify(results, null, 4) }}</pre>
      </div>
    </div>
  </div>
  <div
    v-if="statusType === 'alert-success'"
    role="alert"
    className="alert alert-success"
  >
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="stroke-current shrink-0 h-6 w-6"
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
    <span>{{ statusStr }}</span>
  </div>
  <div
    v-if="statusType === 'alert-error'"
    role="alert"
    className="alert alert-error"
  >
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="stroke-current shrink-0 h-6 w-6"
      fill="none"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
      />
    </svg>
    <span>{{ statusStr }}</span>
  </div>
</template>

<style scoped>
.alert {
  margin-top: 1rem;
}
.center {
  margin: auto;
}
.horizontal {
  display: flex;
  flex-direction: row;
}
.underline {
  text-decoration: underline;
}
h4 {
  margin-bottom: 1em;
}

textarea {
  resize: none;
  padding: auto;
}
#paste-images::placeholder {
  font-size: 1.5rem;
}
.vertical {
  flex-direction: row;
}
.main-run {
  display: flex;
  flex-direction: column;
}
</style>
