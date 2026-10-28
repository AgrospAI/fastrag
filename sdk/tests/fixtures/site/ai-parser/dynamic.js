let batch = 0;
const button = document.querySelector('#more');
button.addEventListener('click', async () => {
  button.disabled = true;
  const articles = await (await fetch('batches.json')).json();
  // Simulate an asynchronous backend; this text is absent from the initial HTML.
  await new Promise(resolve => setTimeout(resolve, 75));
  const article = document.createElement('article');
  const heading = document.createElement('h2');
  heading.textContent = articles[batch].title;
  const paragraph = document.createElement('p');
  paragraph.textContent = articles[batch].text;
  article.append(heading, paragraph);
  document.querySelector('#entries').append(article);
  batch += 1;
  if (batch === articles.length) button.remove();
  else button.disabled = false;
});
