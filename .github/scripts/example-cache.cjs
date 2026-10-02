// Shared by matrix planning and the action that saves newly generated examples.
async function exampleCache(example, prefix, suffix, glob) {
  const patterns = [
    `docs/src/examples/${example}.py`,
    'requirements.txt',
    'requirements-examples.txt',
    'catalogue_docs.py',
    'generate_examples.py',
    'run_example.py',
    '.github/precomputed-examples.json',
    'generate_example_domain.py',
    'generate_domains.py',
  ];
  return {
    key: `${prefix}${suffix}-${example}-${await glob.hashFiles(patterns.join('\n'))}`,
    paths: [
      `docs/public/examples/${example}*`,
      `docs/public/images/examples/${example}*`,
      `docs/public/example-domains/${example}*`,
    ],
  };
}

module.exports = { exampleCache };
