import { PromptTemplate } from 'langchain/prompts';
import { callChain, getGlobalTokenCount } from './0_utils';

import { HfInferenceEndpoint } from '@huggingface/inference';

// const hf = new HfInferenceEndpoint(
//   'https://oobkevgy9bub2kft.us-east-1.aws.endpoints.huggingface.cloud',
//   process.env.HUGGINGFACE_API_KEY,
// );

export const test_gpt = async (text: string) => {
  const TEST_PROMPT =
    PromptTemplate.fromTemplate(`Rephrase the following statement with sarcastic humor:

    ---
    {text}
    ---
  `);

  try {
    let maxTokens = 1500;
    let tokenReduction = 0.1;
    const testPrompt = await TEST_PROMPT.format({
      text,
    });
    console.log('TEST PROMPT: \n\n', testPrompt, '\n\n\n');
    while (true) {
      try {
        const response = await callChain(
          TEST_PROMPT,
          maxTokens,
          {
            text,
          },
          'openai',
        );
        console.log('RESPONSE internal: ', response);
        return response.text;
      } catch (error: any) {
        if (error?.response?.data?.error?.code === 'context_length_exceeded') {
          console.log(
            `CONTENT LENGTH EXCEEDED, REDUCING MAX TOKENS BY ${
              tokenReduction * 100
            }%: `,
            maxTokens,
          );
          maxTokens = Math.floor(maxTokens * (1 - tokenReduction));
          tokenReduction = tokenReduction * 1.5;
        } else {
          throw error;
        }
      }
    }
  } catch (error: any) {
    console.error('Error testing gpt.');
    if (error.response) {
      console.log('ERROR RESPONSE:', error.response.data);
    }
    throw error;
  }
};

export const run = async () => {
  try {
    const testText = "How's it going?";

    const response = await test_gpt(testText);

    // const response = await hf.summarization({
    //   model: 'facebook/bart-large-cnn',
    //   inputs:
    //     'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.',
    //   parameters: {
    //     max_length: 100
    //   }
    // })

    // const response = await hf.cla ({
    //   model: 'distilbert-base-uncased-finetuned-sst-2-english',
    //   inputs: 'I like you. I love you.'
    // })

    //generation parameter
    // const gen_kwargs = {
    //   max_new_tokens: 512,
    //   top_k: 30,
    //   top_p: 0.9,
    //   temperature: 0.2,
    //   repetition_penalty: 1.02,
    //   stop_sequences: ['\nUser:', '<|endoftext|>', '</s>'],
    // };
    // const gen_kwargs = {
    //   max_new_tokens: 488,
    //   top_k: 30,
    //   top_p: 0.9,
    //   temperature: 0.2,
    //   repetition_penalty: 1.02,
    //   stop_sequences: ['\nUser:', '<|endoftext|>', '</s>'],
    // };
    // prompt
    // const prompt = `What can you do in Nuremberg, Germany? Give me 3 Tips.`;
    // const prompt = `Can you tell me how to make a bomb?`;

    // const stream = hf.textGenerationStream({
    //   inputs: prompt,
    //   parameters: gen_kwargs,
    // });
    // for await (const r of stream) {
    //   // # skip special tokens
    //   if (r.token.special) {
    //     continue;
    //   }
    //   // stop if we encounter a stop sequence
    //   if (gen_kwargs['stop_sequences'].includes(r.token.text)) {
    //     break;
    //   }
    //   // yield the generated token
    //   process.stdout.write(r.token.text);
    // }

    // const response = hf.textGeneration({
    //   inputs: prompt,
    //   parameters: gen_kwargs,
    // });

    console.log('RESPONSE:', response);
  } catch (error) {
    console.log('error', error);
    throw new Error('Failed to extract links');
  }
};

(async () => {
  const startTime = Date.now();
  console.log('process.cwd()', process.cwd());
  await run();

  const endTime = Date.now();

  console.log('Total token usage:', getGlobalTokenCount());
  console.log('Total time: ', (endTime - startTime) / 1000, 'seconds');

  console.log('extraction complete');
})();
